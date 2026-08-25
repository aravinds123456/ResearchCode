"""Versioned prompt pack (Algoverse lecture, 23 Aug 2026).

Prompts live in forecasting/prompts/pack.json so teammates load the same
text instead of recreating it. Every seed and tree row stores the pack
version and prompt ids that produced it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(__file__).resolve().parent
PACK_PATH = DIR / "prompts" / "pack.json"
PILOT_PATH = DIR / "results" / "pilot.json"
EXPERIMENT_LOG = DIR / "results" / "experiments.jsonl"
DEFAULT_PILOT_SEEDS = 10
DEFAULT_PILOT_QUESTIONS = 10

_PACK = None
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

PILOT_ERROR = """
Start with ~10 examples to debug prompts before scaling.

    python forecasting/generate_seeds.py --pilot
    python forecasting/pipeline.py tree --pilot --resume

A 100-seed Azure run without that step is the expensive, untraceable
error the Algoverse lecture warned about. Re-run --pilot after any
prompt-pack change. Pass --skip-pilot only if you already finished that
10-example debug.
""".strip()


def load_pack() -> dict:
    global _PACK
    if _PACK is None:
        if not PACK_PATH.exists():
            raise SystemExit(f"Missing prompt pack {PACK_PATH}")
        _PACK = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    return _PACK


def prompt_pack_version() -> int:
    return int(load_pack()["prompt_pack_version"])


def prompt_ids() -> dict[str, str]:
    return {name: spec["id"] for name, spec in load_pack()["prompts"].items()}


def prompt_text(name: str) -> str:
    return load_pack()["prompts"][name]["text"]


def fill_prompt(name: str, **kwargs) -> str:
    """Replace {placeholders} without treating JSON braces as format fields."""
    values = {key: "" if value is None else str(value) for key, value in kwargs.items()}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            return values[key]
        return match.group(0)

    return _PLACEHOLDER.sub(repl, prompt_text(name))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_experiment(kind: str, **fields) -> None:
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": utc_now(),
        "kind": kind,
        "prompt_pack_version": prompt_pack_version(),
        "prompt_ids": prompt_ids(),
        **fields,
    }
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_pilot() -> dict:
    if not PILOT_PATH.exists():
        return {}
    try:
        return json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_pilot_stage(stage: str, **fields) -> dict:
    data = read_pilot()
    data["prompt_pack_version"] = prompt_pack_version()
    data["prompt_ids"] = prompt_ids()
    data["updated_at"] = utc_now()
    data[stage] = {"created_at": utc_now(), **fields}
    PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PILOT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def require_pilot(*, stage: str, n: int, dry_run: bool, skip_pilot: bool) -> None:
    """Block scaling past ~10 examples until a matching prompt-pack pilot exists."""
    if dry_run or skip_pilot or n <= DEFAULT_PILOT_SEEDS:
        return
    data = read_pilot()
    stage_row = data.get(stage) or {}
    if data.get("prompt_pack_version") != prompt_pack_version():
        raise SystemExit(PILOT_ERROR)
    if int(stage_row.get("n") or 0) < DEFAULT_PILOT_SEEDS:
        raise SystemExit(PILOT_ERROR)
