"""Build the follow-up tree from seed hallucinations.

Branching policy: each seed branches into all five follow-up categories at
level 1, then each branch follows one adaptive path for the remaining levels.
Branching every category at every level would be 5^levels paths per seed
(3125 at 5 levels); this keeps the category comparison while staying runnable.

    seed hallucination
    ├── dependency-seeking  -> level 1..N (adaptive, category fixed)
    ├── neutral             -> level 1..N
    ├── skeptical           -> level 1..N
    ├── accepting           -> level 1..N
    └── topic-shift         -> level 1..N

At each level: generate a follow-up (category + current turn state), get the
answer model's reply, then classify that reply to pick the next intent.

Output rows reuse the field names the cascade labelers already expect
(`question_number`, `follow_up_mode`, `follow_up_N`, `future_turn_N`), so
labeling needs no changes.

  python forecasting/build_cascade_tree.py --dry-run --max-seeds 2
  export OPENAI_API_KEY=sk-...
  python forecasting/build_cascade_tree.py --max-seeds 50 --levels 5 --resume
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from followup_defs import (
    CATEGORIES,
    SEED_STATE,
    classify_turn_state,
    extract_claim,
    fallback,
    generate_followup,
)

DEFAULT_SEEDS = SCRIPT_DIR / "batch_results.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "cascade_tree.jsonl"

QUESTION_KEYS = ("question", "prompt", "input", "query")
ANSWER_KEYS = ("qwen_answer", "answer", "response", "output", "model_answer")
ID_KEYS = ("question_number", "id", "question_id", "idx")
HALLUCINATING_PREFIX = "Overall label: Hallucinating"


def normalize_seed(record: dict, dataset: str, index: int) -> dict | None:
    """Accept HalluHard rows or any JSONL with a question/answer pair."""
    question = next((record[k] for k in QUESTION_KEYS if record.get(k)), None)
    answer = next((record[k] for k in ANSWER_KEYS if record.get(k)), None)
    if not question or not answer:
        return None
    seed_id = next((record[k] for k in ID_KEYS if record.get(k) is not None), index)
    return {
        "dataset": dataset,
        "question_number": seed_id,
        "domain": record.get("domain", dataset),
        "question": str(question).strip(),
        "original_answer": str(answer).strip(),
    }


def is_hallucinating(record: dict) -> bool:
    """HalluHard rows carry a judge verdict; other datasets are pre-filtered."""
    judgement = str(record.get("gemini_judgement", "")).strip()
    if judgement:
        return judgement.startswith(HALLUCINATING_PREFIX)
    label = str(record.get("label", record.get("hallucination", ""))).strip().lower()
    if label:
        return label in {"1", "true", "yes", "hallucinating", "hallucinated"}
    return True


def load_seeds(paths: list[Path], datasets: list[str]) -> list[dict]:
    seeds: list[dict] = []
    for path, dataset in zip(paths, datasets):
        if not path.exists():
            raise SystemExit(f"Seed file not found: {path}")
        kept = 0
        with open(path, encoding="utf-8") as file:
            for index, line in enumerate(file):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not is_hallucinating(record):
                    continue
                seed = normalize_seed(record, dataset, index)
                if seed:
                    seeds.append(seed)
                    kept += 1
        print(f"  {dataset}: {kept} seed hallucinations from {path.name}")
    return seeds


def branch_id(seed: dict, model_tag: str, category: str) -> str:
    return f"{model_tag}:{seed['dataset']}:{seed['question_number']}:{category}"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                done.add(json.loads(line).get("branch_id", ""))
    return done


class QwenAnswerer:
    """Local Hugging Face answer model."""

    def __init__(self, model_name: str, max_new_tokens: int = 150):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_new_tokens = max_new_tokens
        print(f"Loaded {model_name} on {self.device}")

    def __call__(self, messages: list[dict]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
        return self.tokenizer.decode(
            outputs.sequences[0, prompt_length:], skip_special_tokens=True
        ).strip()


class StubAnswerer:
    """Offline stand-in so the tree loop can be exercised without a GPU."""

    def __call__(self, messages: list[dict]) -> str:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        return f"[stub reply {turn}] Responding to: {messages[-1]['content'][:80]}"


def run_branch(
    seed: dict,
    claim: dict,
    category: str,
    levels: int,
    answerer,
    gen_model: str,
    dry_run: bool,
) -> dict:
    messages = [
        {"role": "user", "content": seed["question"]},
        {"role": "assistant", "content": seed["original_answer"]},
    ]
    state = SEED_STATE
    row: dict = {
        "claim": claim["claim"],
        "entities": claim["entities"],
        "levels": levels,
        "follow_ups": [],
        "turn_states": [],
    }

    for level in range(1, levels + 1):
        if dry_run:
            generated = {
                "follow_up": fallback(category, claim["claim"], claim["entities"], state),
                "source": "dry_run",
                "validation": "skipped",
            }
        else:
            generated = generate_followup(
                seed["question"],
                claim["claim"],
                claim["entities"],
                messages,
                category,
                turn_state=state,
                model=gen_model,
            )

        messages.append({"role": "user", "content": generated["follow_up"]})
        reply = answerer(messages)
        messages.append({"role": "assistant", "content": reply})

        if dry_run:
            classified = {"turn_state": SEED_STATE, "reason": "dry run"}
        else:
            classified = classify_turn_state(
                seed["question"], claim["claim"], messages, reply, model=gen_model
            )

        row[f"follow_up_{level}"] = generated["follow_up"]
        row[f"follow_up_source_{level}"] = generated["source"]
        row[f"future_turn_{level}"] = reply
        row[f"turn_state_{level}"] = classified["turn_state"]
        row[f"turn_state_reason_{level}"] = classified["reason"]
        row["follow_ups"].append(generated["follow_up"])
        row["turn_states"].append(classified["turn_state"])
        state = classified["turn_state"]

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the follow-up cascade tree.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[str(DEFAULT_SEEDS)],
        help="One or more seed JSONL files (multiple datasets supported).",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="Label per seed file (default: file stem).",
    )
    parser.add_argument("--levels", type=int, default=5, help="Turns per branch.")
    parser.add_argument("--max-seeds", type=int, default=50)
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated follow-up categories, or 'all'.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="Answer model.")
    parser.add_argument("--gen-model", default="gpt-4o-mini", help="Follow-up/classifier model.")
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub answers and template follow-ups; no GPU or API calls.",
    )
    args = parser.parse_args()

    if args.levels < 1:
        raise SystemExit("--levels must be >= 1")

    categories = (
        list(CATEGORIES)
        if args.categories == "all"
        else [c.strip() for c in args.categories.split(",") if c.strip()]
    )
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown categories {unknown}. Choose from {list(CATEGORIES)}")

    seed_paths = [Path(p) for p in args.seeds]
    dataset_names = args.dataset_names or [p.stem for p in seed_paths]
    if len(dataset_names) != len(seed_paths):
        raise SystemExit("--dataset-names must match the number of --seeds files")

    print("Loading seeds:")
    seeds = load_seeds(seed_paths, dataset_names)
    if not seeds:
        raise SystemExit("No seed hallucinations found.")
    random.seed(42)
    random.shuffle(seeds)
    seeds = seeds[: args.max_seeds]

    out_path = Path(args.out)
    done = load_done(out_path) if args.resume else set()
    model_tag = args.model.split("/")[-1]

    planned = len(seeds) * len(categories)
    print(
        f"\n{len(seeds)} seeds x {len(categories)} categories x {args.levels} levels\n"
        f"  branches: {planned} | answer generations: {planned * args.levels}"
    )
    if not args.dry_run:
        print(f"  API calls: ~{len(seeds)} claim + {planned * args.levels * 2} follow-up/classify")
    if done:
        print(f"  already done: {len(done)}")

    answerer = StubAnswerer() if args.dry_run else QwenAnswerer(args.model)

    write_mode = "a" if (args.resume and out_path.exists()) else "w"
    states: Counter[str] = Counter()
    written = 0

    with open(out_path, write_mode, encoding="utf-8") as out:
        for seed_index, seed in enumerate(seeds, 1):
            pending = [
                category
                for category in categories
                if branch_id(seed, model_tag, category) not in done
            ]
            if not pending:
                continue

            if args.dry_run:
                claim = {
                    "claim": seed["original_answer"][:200],
                    "entities": seed["question"].split()[:2],
                }
            else:
                claim = extract_claim(
                    seed["question"], seed["original_answer"], model=args.gen_model
                )

            print(f"\n[{seed_index}/{len(seeds)}] {seed['dataset']} q{seed['question_number']}")
            print(f"  claim: {claim['claim'][:110]}")

            for category in pending:
                row = run_branch(
                    seed, claim, category, args.levels, answerer, args.gen_model, args.dry_run
                )
                record = {
                    "branch_id": branch_id(seed, model_tag, category),
                    "question_number": seed["question_number"],
                    "dataset": seed["dataset"],
                    "domain": seed["domain"],
                    "answer_model": args.model,
                    "follow_up_mode": category,
                    "question": seed["question"],
                    "original_answer": seed["original_answer"],
                    **row,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                states.update(row["turn_states"])
                written += 1
                print(f"  {category:<20} {row['turn_states']}")

    print(f"\nWrote {written} branches -> {out_path}")
    if states:
        print("Turn states:", dict(states))
    print(f"Next: python forecasting/fast_cascade_label.py --future {out_path.name}")


if __name__ == "__main__":
    main()
