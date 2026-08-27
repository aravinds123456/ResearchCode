"""JSON parsing that fails closed. Parse failure is never a scientific label."""

from __future__ import annotations

import json
import re
from typing import Any

from libs.json_utils import extract_json_from_response, sanitize_json_string


class ParseError(ValueError):
    """Raised when model output cannot be parsed as the expected JSON object."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ParseError("empty model output")
    candidates = [raw]
    fenced = _FENCE.search(raw)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    try:
        extracted = extract_json_from_response(raw)
        if extracted and extracted not in candidates:
            candidates.insert(0, extracted)
    except Exception:
        pass
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(sanitize_json_string(candidate))
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(sanitize_json_string(candidate[start : end + 1]))
                except (json.JSONDecodeError, ValueError) as nested:
                    last_error = nested
                    continue
            else:
                continue
        if isinstance(payload, dict):
            return payload
        last_error = ParseError(f"JSON value is {type(payload).__name__}, not object")
    raise ParseError(f"unparseable JSON: {last_error}")


def parse_json_list(text: str) -> list[Any]:
    raw = (text or "").strip()
    if not raw:
        raise ParseError("empty model output")
    try:
        extracted = extract_json_from_response(raw)
    except Exception:
        extracted = raw
    start_list = extracted.find("[")
    start_obj = extracted.find("{")
    if start_list >= 0 and (start_obj < 0 or start_list < start_obj):
        end = extracted.rfind("]")
        blob = extracted[start_list : end + 1] if end > start_list else extracted
        payload = json.loads(sanitize_json_string(blob))
        if isinstance(payload, list):
            return payload
    obj = parse_json_object(raw)
    for key in ("claims", "candidates", "items"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise ParseError("JSON object did not contain a claims list")
