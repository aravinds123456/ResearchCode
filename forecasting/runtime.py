"""Model loading and judge/follow-up API calls."""

from __future__ import annotations

import os
import time

from cascade import DEFAULT_OPENAI_JUDGE, env_int, env_str, strip_thinking

GEMINI_RETRIES = env_int("GEMINI_RETRIES", 5)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 400)
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
    print(f"Loading {name} on {device} ({dtype})...")
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


def build_model_inputs(messages):
    kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    # Qwen3.5 otherwise spends the token budget on a hidden chain-of-thought.
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


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


def load_qwen(name: str):
    """Back-compat chat callable used by the cascade tree."""
    init_model(name)

    def chat(messages):
        return generate_response(messages, max_new_tokens=max(256, MAX_NEW_TOKENS // 2))

    print(f"Loaded {name} on {device}")
    return chat


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
