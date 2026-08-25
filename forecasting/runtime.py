"""Model loading and judge/follow-up API calls."""

from __future__ import annotations

import os
import sys
import time

from cascade import (
    DEFAULT_OPENAI_JUDGE,
    DEFAULT_TEST_MODEL,
    ENABLE_THINKING,
    env_int,
    env_str,
    strip_thinking,
)

GEMINI_RETRIES = env_int("GEMINI_RETRIES", 5)
# Hidden GPT-OSS reasoning counts against this. Default is the usual Azure
# GPT-OSS output cap. Override with MAX_TOKENS or MAX_NEW_TOKENS.
DEFAULT_MAX_TOKENS = 32768
MAX_NEW_TOKENS = env_int("MAX_TOKENS", env_int("MAX_NEW_TOKENS", DEFAULT_MAX_TOKENS))
JUDGE_MODEL_NAME = env_str("JUDGE_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = env_str("OPENAI_LABEL_MODEL", DEFAULT_OPENAI_JUDGE)
OPENAI_REASONING_EFFORT = env_str("OPENAI_REASONING_EFFORT", "minimal")

tokenizer = None
model = None
device = None
_gemini_client = None
_gemini_model = None


def resolve_device():
    import torch

    requested = env_str("DEVICE", "")
    if requested:
        return torch.device(requested)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(target_device):
    import torch

    requested = env_str("TORCH_DTYPE", "auto")
    if requested != "auto":
        return getattr(torch, requested)
    if target_device.type == "cpu":
        return torch.float32
    return torch.float16


def init_model(name: str):
    global tokenizer, model, device
    if model is not None:
        return tokenizer, model, device

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device()
    dtype = resolve_dtype(device)
    trust = env_str("TRUST_REMOTE_CODE", "0") == "1"
    print(f"Loading {name} on {device} ({dtype}), thinking={'on' if ENABLE_THINKING else 'off'}...")
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, trust_remote_code=trust,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        print(f"Warning: {name} has no chat template; using role-prefixed formatting.")
    return tokenizer, model, device


_THINKING_MODE_LOGGED = False


def with_no_think_tag(messages):
    """Copy messages and append Qwen's /no_think soft switch to the last user turn."""
    copied = [{**message} for message in messages]
    for message in reversed(copied):
        if message.get("role") == "user":
            content = str(message.get("content") or "").rstrip()
            if "/no_think" not in content:
                message["content"] = f"{content}\n/no_think"
            break
    return copied


def _log_thinking_mode(how: str) -> None:
    global _THINKING_MODE_LOGGED
    if _THINKING_MODE_LOGGED:
        return
    _THINKING_MODE_LOGGED = True
    state = "on" if ENABLE_THINKING else "off"
    print(f"Qwen reasoning/thinking: {state} ({how})")


def build_model_inputs(messages):
    if tokenizer is None:
        raise RuntimeError("Call init_model() before build_model_inputs().")
    kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if ENABLE_THINKING:
        _log_thinking_mode("ENABLE_THINKING=1")
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    # Qwen3.5 thinks unless the chat template gets an explicit hard switch.
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        _log_thinking_mode("enable_thinking=False")
        return encoded
    except TypeError:
        pass
    try:
        encoded = tokenizer.apply_chat_template(
            messages, chat_template_kwargs={"enable_thinking": False}, **kwargs
        )
        _log_thinking_mode("chat_template_kwargs enable_thinking=False")
        return encoded
    except TypeError:
        pass
    _log_thinking_mode("/no_think fallback; tokenizer has no enable_thinking argument")
    return tokenizer.apply_chat_template(with_no_think_tag(messages), **kwargs)


def generate_response(messages, max_new_tokens: int | None = None) -> str:
    import torch

    if tokenizer is None or model is None:
        raise RuntimeError("Call init_model() before generate_response().")
    model_inputs = build_model_inputs(messages)
    model_inputs = {key: value.to(device) for key, value in model_inputs.items() if hasattr(value, "to")}
    input_length = model_inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
    generated_tokens = outputs.sequences[0, input_length:]
    return strip_thinking(tokenizer.decode(generated_tokens, skip_special_tokens=True))


def followup_max_new_tokens() -> int:
    """Token budget for tree follow-ups. Hidden GPT-OSS reasoning counts against this."""
    override = os.environ.get("FOLLOWUP_MAX_NEW_TOKENS", "").strip()
    if override:
        return int(override)
    return MAX_NEW_TOKENS


def empty_retry_tokens(current: int) -> int:
    override = os.environ.get("AZURE_EMPTY_RETRY_TOKENS", "").strip()
    if override:
        return int(override)
    return min(max(int(current) * 2, 4096), DEFAULT_MAX_TOKENS)


def load_qwen(name: str):
    """Local HuggingFace chat callable (Qwen and other HF models)."""
    init_model(name)

    def chat(messages):
        return generate_response(messages, max_new_tokens=followup_max_new_tokens())

    print(f"Loaded {name} on {device}")
    return chat


def azure_credentials() -> tuple[str, str]:
    endpoint = env_str("AZURE_OPENAI_ENDPOINT", env_str("AZURE_ENDPOINT", ""))
    key = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_API_KEY")
        or ""
    ).strip()
    return endpoint.rstrip("/"), key


def uses_azure_answer(name: str) -> bool:
    backend = env_str("ANSWER_BACKEND", "").lower()
    if backend in {"azure", "azure-openai", "api"}:
        return True
    if backend in {"local", "hf", "huggingface", "qwen"}:
        return False
    lowered = (name or "").lower()
    if "gpt-oss" in lowered or "gptoss" in lowered:
        return True
    endpoint, key = azure_credentials()
    return bool(endpoint and key)


def azure_deployment(name: str) -> str:
    return env_str("AZURE_OPENAI_DEPLOYMENT", name)


def list_azure_deployments() -> list[str]:
    """Names shown in Azure Portal → Deployments for this resource."""
    import json
    import urllib.error
    import urllib.request

    endpoint, key = azure_credentials()
    if not endpoint or not key:
        return []
    version = env_str("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    url = f"{endpoint}/openai/deployments?api-version={version}"
    request = urllib.request.Request(url, headers={"api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    names = []
    for item in payload.get("data") or payload.get("value") or []:
        name = item.get("id") or item.get("name")
        if name:
            names.append(str(name))
    return names


def is_deployment_missing(error: BaseException) -> bool:
    text = str(error)
    code = ""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        code = str(inner.get("code") or "")
    return "DeploymentNotFound" in text or code == "DeploymentNotFound"


def deployment_missing_message(deployment: str, found: list[str] | None = None) -> str:
    names = found if found is not None else list_azure_deployments()
    listed = ", ".join(names) if names else "(none listed; open Azure Portal → Deployments)"
    return (
        f"Azure has no deployment named {deployment!r} on this resource. "
        "That is not a missing second API key.\n"
        "TEST_MODEL defaults to gpt-oss-20b; the name in the portal is often "
        "different (gpt-oss-120b, or whatever you typed when you deployed).\n"
        f"Deployments on this resource: {listed}\n"
        "Copy the exact name, then:\n"
        "  export AZURE_OPENAI_DEPLOYMENT=<exact-name>\n"
        "Stop any tree that started after generate_seeds crashed; it is not using GPT-OSS seeds."
    )


def _attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def content_to_text(content) -> str:
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


def extract_visible_text(message) -> str:
    """Public assistant text only. Hidden GPT-OSS reasoning is not a visible answer."""
    return content_to_text(_attr(message, "content"))


def extract_reasoning_text(message) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = _attr(message, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def describe_completion(choice, usage) -> str:
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


def azure_reasoning_effort() -> str:
    return env_str("AZURE_REASONING_EFFORT", "low")


def azure_send_temperature() -> bool:
    return env_str("AZURE_SEND_TEMPERATURE", "").lower() in {"1", "true", "yes", "on"}


def azure_use_reasoning_fallback() -> bool:
    return env_str("AZURE_USE_REASONING_FALLBACK", "1").lower() not in {"0", "false", "no", "off"}


def azure_create_kwargs(
    deployment: str,
    messages,
    max_new_tokens: int,
    *,
    temperature: float | None = None,
    send_temperature: bool = False,
    reasoning_effort: str | None = None,
) -> dict:
    kwargs = {
        "model": deployment,
        "messages": messages,
        "max_tokens": int(max_new_tokens),
    }
    if send_temperature and temperature is not None:
        kwargs["temperature"] = temperature
    effort = (reasoning_effort or "").strip()
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


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


def azure_chat_create(client, kwargs: dict):
    """Create a chat completion, stripping parameters Azure GPT-OSS rejects."""
    pending = dict(kwargs)
    stripped: set[str] = set()
    extra_body_attempted = False
    while True:
        try:
            return client.chat.completions.create(**pending)
        except TypeError:
            if extra_body_attempted or "reasoning_effort" not in pending:
                raise
            extra_body_attempted = True
            effort = pending.pop("reasoning_effort")
            extra = dict(pending.get("extra_body") or {})
            extra["reasoning_effort"] = effort
            pending["extra_body"] = extra
        except Exception as error:
            if is_deployment_missing(error):
                raise SystemExit(deployment_missing_message(str(pending.get("model", "")))) from error
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


def _azure_answer_once(client, kwargs: dict):
    response = azure_chat_create(client, kwargs)
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = _attr(choice, "message") if choice is not None else None
    usage = _attr(response, "usage")
    visible = extract_visible_text(message) if message is not None else ""
    return choice, message, usage, visible


def azure_answer_text(
    client,
    messages,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    model_name: str | None = None,
    send_temperature: bool | None = None,
    reasoning_effort: str | None = None,
    use_reasoning_fallback: bool | None = None,
) -> str:
    """One Azure completion with an empty-content retry. Does not call Azure itself except via client."""
    name = model_name or env_str("TEST_MODEL", DEFAULT_TEST_MODEL)
    deployment = azure_deployment(name)
    if send_temperature is None:
        send_temperature = azure_send_temperature()
    if reasoning_effort is None:
        reasoning_effort = azure_reasoning_effort()
    if use_reasoning_fallback is None:
        use_reasoning_fallback = azure_use_reasoning_fallback()
    kwargs = azure_create_kwargs(
        deployment,
        messages,
        max_new_tokens,
        temperature=temperature,
        send_temperature=send_temperature,
        reasoning_effort=reasoning_effort,
    )
    choice, message, usage, visible = _azure_answer_once(client, kwargs)
    if visible:
        return strip_thinking(visible)

    why = describe_completion(choice, usage)
    retry_tokens = empty_retry_tokens(max_new_tokens)
    if retry_tokens > int(max_new_tokens):
        print(
            f"empty assistant content ({why}); retrying with max_tokens={retry_tokens}",
            file=sys.stderr,
        )
        retry_kwargs = dict(kwargs)
        if "max_tokens" in retry_kwargs:
            retry_kwargs["max_tokens"] = retry_tokens
        else:
            retry_kwargs["max_completion_tokens"] = retry_tokens
        choice, message, usage, visible = _azure_answer_once(client, retry_kwargs)
        if visible:
            return strip_thinking(visible)
        why = describe_completion(choice, usage)
    else:
        print(f"empty assistant content ({why})", file=sys.stderr)

    why = describe_completion(choice, usage)
    reasoning = extract_reasoning_text(message) if message is not None else ""
    if reasoning and use_reasoning_fallback:
        print(
            f"empty assistant content after retry ({why}); "
            "using reasoning_content as last resort "
            "(hidden chain-of-thought is not a normal visible answer)",
            file=sys.stderr,
        )
        return strip_thinking(reasoning)
    print(f"empty assistant content after retry ({why})", file=sys.stderr)
    return ""


def make_azure_client():
    from openai import AzureOpenAI, OpenAI

    endpoint, key = azure_credentials()
    if not endpoint or not key:
        raise SystemExit(
            "GPT-OSS on Azure needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "(or AZURE_ENDPOINT / AZURE_API_KEY)."
        )
    if "/openai/v1" in endpoint or endpoint.endswith("/models"):
        base = endpoint if endpoint.endswith("/") else f"{endpoint}/"
        return OpenAI(base_url=base, api_key=key)
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=env_str("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def answer_chat(
    messages,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
    model_name: str | None = None,
) -> str:
    """One answering-model completion: Azure GPT-OSS or local HF."""
    name = model_name or env_str("TEST_MODEL", DEFAULT_TEST_MODEL)
    tokens = max_new_tokens if max_new_tokens is not None else followup_max_new_tokens()
    if not uses_azure_answer(name):
        init_model(name)
        return generate_response(messages, max_new_tokens=tokens)

    return azure_answer_text(
        make_azure_client(),
        messages,
        tokens,
        temperature=temperature,
        model_name=name,
    )


def load_answer_model(name: str):
    """Chat callable for the cascade tree: Azure GPT-OSS by default."""
    if uses_azure_answer(name):
        endpoint, _ = azure_credentials()
        print(f"Answer model: {azure_deployment(name)} via Azure ({endpoint or 'set AZURE_OPENAI_ENDPOINT'})")
        print(f"Follow-up max_tokens={followup_max_new_tokens()} reasoning_effort={azure_reasoning_effort() or 'off'}")

        def chat(messages):
            return answer_chat(
                messages,
                max_new_tokens=followup_max_new_tokens(),
                temperature=0.0,
                model_name=name,
            )

        return chat
    return load_qwen(name)


def _uses_responses_api(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(("gpt-5", "o1", "o3", "o4"))


def _openai_text(client, prompt: str, as_json: bool) -> str:
    """GPT-5 rejects temperature=0 on chat completions; use the Responses API."""
    if _uses_responses_api(OPENAI_MODEL):
        kwargs = {
            "model": OPENAI_MODEL,
            "input": prompt,
            "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        }
        if as_json:
            kwargs["text"] = {"format": {"type": "json_object"}}
        return (client.responses.create(**kwargs).output_text or "").strip()
    extra = {"response_format": {"type": "json_object"}} if as_json else {}
    return (
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **extra,
        ).choices[0].message.content
        or ""
    ).strip()


def gpt(prompt: str, as_json: bool = True):
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("Set OPENAI_API_KEY")
    reply = _openai_text(OpenAI(), prompt, as_json=as_json)
    if as_json:
        import json
        return json.loads(reply or "{}")
    return reply


def active_judge_model() -> str:
    if judge_backend() == "gemini":
        return JUDGE_MODEL_NAME
    return OPENAI_MODEL


def setup_gemini():
    global _gemini_client, _gemini_model
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY")

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_model = JUDGE_MODEL_NAME
        return "google-genai"
    except Exception:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(JUDGE_MODEL_NAME)
        _gemini_client = "legacy"
        return "google.generativeai"


def call_gemini(prompt: str) -> str:
    if _gemini_model is None:
        setup_gemini()
    last_error = None
    for attempt in range(GEMINI_RETRIES):
        try:
            if _gemini_client == "legacy":
                response = _gemini_model.generate_content(prompt)
                text = getattr(response, "text", None)
            else:
                response = _gemini_client.models.generate_content(model=_gemini_model, contents=prompt)
                text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty response")
            return text.strip()
        except Exception as error:
            last_error = error
            if attempt == GEMINI_RETRIES - 1:
                raise RuntimeError(f"Gemini call failed after {GEMINI_RETRIES} attempts") from error
            wait_seconds = 2 ** attempt
            print(f"Gemini retry {attempt + 1}/{GEMINI_RETRIES} after error: {error}")
            time.sleep(wait_seconds)
    raise RuntimeError(str(last_error))


def judge_backend() -> str:
    requested = env_str("JUDGE_BACKEND", "")
    if requested:
        return requested.lower()
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "openai"
