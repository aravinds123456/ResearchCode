"""Azure OpenAI SamplerBase adapter.

Provider details (endpoint, deployment, token-parameter compatibility,
visible-output extraction, reasoning-effort handling) stay here so callers
never construct AzureOpenAI/AsyncAzureOpenAI themselves.

GPT-OSS and other chat-completions models use Chat Completions.
GPT-5 / o-series models use the Responses API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any

import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from libs.types import MessageList, SamplerBase, SamplerResponse

import dotenv

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

_shared_azure_client: AsyncAzureOpenAI | AsyncOpenAI | None = None
_shared_azure_client_key: tuple[str, str, str] | None = None

DEFAULT_MAX_TOKENS = 32768
DEFAULT_API_VERSION = "2024-12-01-preview"


def azure_credentials() -> tuple[str, str]:
    endpoint = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_ENDPOINT")
        or ""
    ).strip().rstrip("/")
    key = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_API_KEY")
        or ""
    ).strip()
    return endpoint, key


def azure_api_version() -> str:
    return (os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION).strip()


def uses_responses_api(model: str) -> bool:
    lowered = (model or "").lower()
    return lowered.startswith(("gpt-5", "o1", "o3", "o4"))


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text).strip())
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_visible_text(message: Any) -> str:
    """Public assistant text only. Hidden GPT-OSS reasoning is not a visible answer."""
    return content_to_text(_attr(message, "content"))


def extract_reasoning_text(message: Any) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = _attr(message, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def describe_completion(choice: Any, usage: Any) -> str:
    finish = _attr(choice, "finish_reason") or "unknown"
    completion = _attr(usage, "completion_tokens")
    details = _attr(usage, "completion_tokens_details")
    reasoning = _attr(details, "reasoning_tokens") if details is not None else None
    parts = [f"finish_reason={finish}"]
    if completion is not None:
        parts.append(f"completion_tokens={completion}")
    if reasoning is not None:
        parts.append(f"reasoning_tokens={reasoning}")
    return ", ".join(parts)


def is_unsupported_parameter(error: BaseException, param: str) -> bool:
    text = str(error).lower()
    if param.lower() not in text:
        return False
    markers = (
        "unsupported",
        "invalid",
        "unknown",
        "not supported",
        "unexpected",
        "does not support",
    )
    return any(marker in text for marker in markers)


def empty_retry_tokens(current: int) -> int:
    override = os.environ.get("AZURE_EMPTY_RETRY_TOKENS", "").strip()
    if override:
        return int(override)
    return min(max(int(current) * 2, 4096), DEFAULT_MAX_TOKENS)


def _is_models_as_a_service(endpoint: str) -> bool:
    return (
        "/openai/v1" in endpoint
        or endpoint.endswith("/models")
        or "/managed-deployments/" in endpoint
    )


def get_shared_azure_client(
    endpoint: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    max_connections: int = 50,
) -> AsyncAzureOpenAI | AsyncOpenAI:
    """Shared Azure client with connection pooling."""
    global _shared_azure_client, _shared_azure_client_key
    resolved_endpoint, resolved_key = azure_credentials()
    endpoint = (endpoint or resolved_endpoint).rstrip("/")
    api_key = api_key or resolved_key
    api_version = api_version or azure_api_version()
    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure sampler needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "(or AZURE_ENDPOINT / AZURE_API_KEY). OPENAI_API_KEY is not used."
        )
    cache_key = (endpoint, api_key, api_version)
    if _shared_azure_client is not None and _shared_azure_client_key == cache_key:
        return _shared_azure_client

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections // 2,
        ),
        timeout=httpx.Timeout(300.0, connect=60.0),
        http1=True,
        http2=False,
    )
    if _is_models_as_a_service(endpoint):
        base = endpoint if endpoint.endswith("/") else f"{endpoint}/"
        _shared_azure_client = AsyncOpenAI(
            base_url=base,
            api_key=api_key,
            timeout=300.0,
            max_retries=0,
            http_client=http_client,
        )
    else:
        _shared_azure_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=300.0,
            max_retries=0,
            http_client=http_client,
        )
    _shared_azure_client_key = cache_key
    _logger.debug("Created shared Azure OpenAI client (endpoint=%s)", endpoint)
    return _shared_azure_client


def _extract_token_usage(response: Any) -> dict[str, int]:
    token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    usage = getattr(response, "usage", None)
    if not usage:
        return token_usage
    token_usage["input_tokens"] = getattr(usage, "input_tokens", None) or getattr(
        usage, "prompt_tokens", 0
    ) or 0
    token_usage["output_tokens"] = getattr(usage, "output_tokens", None) or getattr(
        usage, "completion_tokens", 0
    ) or 0
    token_usage["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None) or getattr(
        usage, "completion_tokens_details", None
    )
    if input_details:
        token_usage["cached_tokens"] = getattr(input_details, "cached_tokens", 0) or 0
    if output_details:
        token_usage["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", 0) or 0
    return token_usage


def _messages_to_responses_input(message_list: MessageList) -> list[dict[str, Any]]:
    packed = []
    for message in message_list:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            packed.append({"role": "developer", "content": content})
        else:
            packed.append({"role": role, "content": content})
    return packed


async def azure_chat_create(client: Any, kwargs: dict) -> Any:
    """Create a chat completion, stripping parameters Azure GPT-OSS rejects."""
    pending = dict(kwargs)
    stripped: set[str] = set()
    extra_body_attempted = False
    while True:
        try:
            return await client.chat.completions.create(**pending)
        except TypeError:
            if extra_body_attempted or "reasoning_effort" not in pending:
                raise
            extra_body_attempted = True
            effort = pending.pop("reasoning_effort")
            extra = dict(pending.get("extra_body") or {})
            extra["reasoning_effort"] = effort
            pending["extra_body"] = extra
        except Exception as error:
            if "temperature" in pending and "temperature" not in stripped and is_unsupported_parameter(
                error, "temperature"
            ):
                pending.pop("temperature", None)
                stripped.add("temperature")
                continue
            if (
                "reasoning_effort" in pending
                and "reasoning_effort" not in stripped
                and is_unsupported_parameter(error, "reasoning_effort")
            ):
                pending.pop("reasoning_effort", None)
                stripped.add("reasoning_effort")
                continue
            extra = pending.get("extra_body")
            if (
                isinstance(extra, dict)
                and "reasoning_effort" in extra
                and "reasoning_effort" not in stripped
                and is_unsupported_parameter(error, "reasoning_effort")
            ):
                extra.pop("reasoning_effort", None)
                stripped.add("reasoning_effort")
                continue
            if "max_tokens" in pending and "max_tokens" not in stripped and (
                is_unsupported_parameter(error, "max_tokens")
                or "max_tokens" in str(error).lower()
            ):
                tokens = pending.pop("max_tokens")
                pending["max_completion_tokens"] = tokens
                stripped.add("max_tokens")
                continue
            if "max_completion_tokens" in pending and "max_completion_tokens" not in stripped and (
                is_unsupported_parameter(error, "max_completion_tokens")
                or "max_completion_tokens" in str(error).lower()
            ):
                tokens = pending.pop("max_completion_tokens")
                pending["max_tokens"] = tokens
                stripped.add("max_completion_tokens")
                continue
            raise


class AzureOpenAISampler(SamplerBase):
    """SamplerBase implementation backed by Azure OpenAI.

    Does not require OPENAI_API_KEY. Serper remains a separate credential.
    """

    def __init__(
        self,
        model: str,
        deployment: str | None = None,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 10,
        websearch: bool = False,
        send_temperature: bool = False,
        use_reasoning_fallback: bool = False,
        api: str | None = None,
    ):
        if websearch:
            raise ValueError(
                "AzureOpenAISampler does not use OpenAI web_search. "
                "Grounding must go through HalluHard/Serper, not a websearch fallback."
            )
        endpoint, key = azure_credentials()
        if not endpoint or not key:
            raise RuntimeError(
                "Azure sampler needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
                "OPENAI_API_KEY is not used for this path."
            )
        self.client = get_shared_azure_client(endpoint, key)
        self.model = model
        self.deployment = deployment or model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.send_temperature = send_temperature
        self.use_reasoning_fallback = use_reasoning_fallback
        if api in {"responses", "chat"}:
            self._api = api
        else:
            self._api = "responses" if uses_responses_api(model) else "chat"
        tag_parts = ["azure", self.deployment]
        if reasoning_effort:
            tag_parts.append(reasoning_effort)
        self._log_tag = "-".join(tag_parts)

    def _chat_kwargs(self, messages: MessageList) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)
        if self.send_temperature and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        effort = (self.reasoning_effort or "").strip()
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs

    async def _chat_once(self, kwargs: dict[str, Any]) -> tuple[Any, Any, Any, str]:
        response = await azure_chat_create(self.client, kwargs)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = _attr(choice, "message") if choice is not None else None
        usage = _attr(response, "usage")
        visible = extract_visible_text(message) if message is not None else ""
        return choice, message, usage, visible

    async def _call_chat(self, message_list: MessageList) -> SamplerResponse:
        kwargs = self._chat_kwargs(message_list)
        choice, message, usage, visible = await self._chat_once(kwargs)
        if not visible and self.max_tokens is not None:
            retry_tokens = empty_retry_tokens(self.max_tokens)
            if retry_tokens > int(self.max_tokens):
                why = describe_completion(choice, usage)
                _logger.warning(
                    "[%s] empty assistant content (%s); retrying with max_tokens=%s",
                    self._log_tag,
                    why,
                    retry_tokens,
                )
                retry_kwargs = dict(kwargs)
                if "max_tokens" in retry_kwargs:
                    retry_kwargs["max_tokens"] = retry_tokens
                else:
                    retry_kwargs["max_completion_tokens"] = retry_tokens
                choice, message, usage, visible = await self._chat_once(retry_kwargs)
        if not visible:
            why = describe_completion(choice, usage)
            reasoning = extract_reasoning_text(message) if message is not None else ""
            if reasoning and self.use_reasoning_fallback:
                _logger.warning(
                    "[%s] empty assistant content (%s); using reasoning_content as last resort",
                    self._log_tag,
                    why,
                )
                visible = reasoning
            else:
                _logger.warning("[%s] empty assistant content (%s)", self._log_tag, why)
        token_usage = _extract_token_usage(
            type("Resp", (), {"usage": usage})()
        )
        if usage is not None and token_usage["total_tokens"] == 0:
            token_usage["input_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
            token_usage["output_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                token_usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0) or 0
        return SamplerResponse(
            response_text=visible,
            response_metadata={
                "usage": usage,
                "backend": "azure",
                "deployment": self.deployment,
                "api": "chat",
                "empty": not bool(visible),
                "finish": describe_completion(choice, usage) if choice is not None else "",
            },
            actual_queried_message_list=message_list,
            token_usage=token_usage,
        )

    async def _call_responses(self, message_list: MessageList) -> SamplerResponse:
        packed = _messages_to_responses_input(message_list)
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "input": packed,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        elif self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        response = await self.client.responses.create(**kwargs)
        if getattr(response, "status", None) == "incomplete":
            response_text = response.output_text or ""
            _logger.warning(
                "[%s] Responses API incomplete: %s",
                self._log_tag,
                getattr(response, "incomplete_details", None),
            )
        else:
            response_text = response.output_text or ""
        return SamplerResponse(
            response_text=(response_text or "").strip(),
            response_metadata={
                "usage": getattr(response, "usage", None),
                "status": getattr(response, "status", None),
                "backend": "azure",
                "deployment": self.deployment,
                "api": "responses",
            },
            actual_queried_message_list=message_list,
            token_usage=_extract_token_usage(response),
        )

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [{"role": "system", "content": self.system_message}] + list(message_list)
        trial = 0
        while True:
            try:
                await asyncio.sleep(random.uniform(0, 0.2))
                if self._api == "responses":
                    return await self._call_responses(message_list)
                return await self._call_chat(message_list)
            except openai.BadRequestError as e:
                _logger.warning("[%s] Bad Request Error: %s", self._log_tag, e)
                raise RuntimeError(f"Azure OpenAI BadRequestError: {e}") from e
            except openai.RateLimitError as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"Azure OpenAI rate limit after {self.max_retries} retries: {e}"
                    ) from e
                backoff = 2**trial + random.uniform(0, 0.5 * (2**trial))
                _logger.debug("[%s] Rate limit, retrying after %.1fs: %s", self._log_tag, backoff, e)
                await asyncio.sleep(backoff)
                trial += 1
            except (openai.APITimeoutError, asyncio.TimeoutError, openai.APIConnectionError) as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"Azure OpenAI connection/timeout after {self.max_retries} retries: {e}"
                    ) from e
                backoff = 2**trial + random.uniform(0, 0.5 * (2**trial))
                await asyncio.sleep(backoff)
                trial += 1
            except Exception as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"Azure OpenAI error after {self.max_retries} retries: {e}"
                    ) from e
                backoff = 2**trial + random.uniform(0, 0.5 * (2**trial))
                _logger.debug(
                    "[%s] API error, retrying after %.1fs: %s: %s",
                    self._log_tag,
                    backoff,
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(backoff)
                trial += 1
