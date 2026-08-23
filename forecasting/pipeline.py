"""Cascade pipeline: HalluHard 5-strategy tree + HallucinationResearchTest labels.

  python forecasting/pipeline.py answer --domain all --resume
  python forecasting/pipeline.py judge  --domain all --resume
  python forecasting/pipeline.py tree   --max-seeds 100 --levels 3 --resume
  python forecasting/pipeline.py label  --resume
  python forecasting/pipeline.py report --from-partial
  python forecasting/pipeline.py tree --dry-run --max-seeds 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from cascade import (
    BATCH,
    CATS,
    DEFAULT_MAX_SEEDS,
    DEFAULT_TEST_MODEL,
    DEFAULT_TURNS,
    ENABLE_THINKING,
    DOMAINS,
    HALL,
    LABELS,
    OUTCOMES,
    P_CLAIM,
    P_DRAFT,
    P_JUDGE,
    P_LABEL,
    P_TURN,
    SCHEMA_VERSION,
    SEED_LABEL,
    STATES,
    TREE,
    FOLLOWUP_TYPE_DESCRIPTIONS,
    backup,
    branch_id,
    check,
    derive_branch_outcome,
    display_state,
    domain_of,
    env_str,
    first_present_field,
    hallucinating,
    hint_for,
    history,
    names,
    normalize_outcome,
    parse_judge_label,
    rows,
    sample_seeds,
    sampling_plan,
    seed_identifier,
    strip_question_prefix,
    strip_thinking,
    write,
)

QWEN = env_str("TEST_MODEL", env_str("QWEN_MODEL", DEFAULT_TEST_MODEL))


def gpt(prompt: str, as_json: bool = True):
    from runtime import gpt as _gpt
    return _gpt(prompt, as_json=as_json)


def load_qwen(name: str):
    from runtime import load_qwen as _load
    return _load(name)


def cmd_answer(args) -> None:
    """Step A: generate turn-0 answers."""
    chat = load_qwen(args.model)
    for domain in (DOMAINS if args.domain == "all" else [args.domain]):
        path, key, offset = DOMAINS[domain]
        out = DIR / f"batch_results_{domain}.jsonl"
        done = {r["question_number"] for r in rows(out)} if args.resume else set()
        seen = bool(done)
        for i, line in enumerate(open(path, encoding="utf-8")):
            if args.n and i >= args.n:
                break
            if offset + i in done:
                continue
            question = json.loads(line)[key].strip()
            answer = strip_question_prefix(question, chat([{"role": "user", "content": question}]))
            write(
                out,
                {
                    "question_number": offset + i,
                    "domain": domain,
                    "question": question,
                    "qwen_answer": answer,
                    "model_answer": answer,
                    "model_name": args.model,
                    "gemini_judgement": "",
                    "schema_version": SCHEMA_VERSION,
                },
                seen,
            )
            seen = True
            print(f"{domain} {offset + i}")


def cmd_judge(args) -> None:
    """Step B: mark each turn-0 answer hallucinating or not, then merge domains."""
    from runtime import active_judge_model
    merged = {r["question_number"]: r for r in rows(BATCH)} if args.resume else {}
    print(f"Judge model: {active_judge_model()}")
    for domain in (DOMAINS if args.domain == "all" else [args.domain]):
        for row in rows(DIR / f"batch_results_{domain}.jsonl"):
            if row["question_number"] in merged and merged[row["question_number"]].get("gemini_judgement"):
                continue
            answer = strip_question_prefix(row["question"], row.get("qwen_answer") or row.get("model_answer", ""))
            verdict = str(gpt(P_JUDGE.format(q=row["question"][:2000], a=answer[:6000]), False)).strip()
            if not verdict.startswith("Overall label:"):
                verdict = (
                    f"Overall label: {'Hallucinating' if 'Hallucinating' in verdict else 'Not Hallucinating'}\n{verdict}"
                )
            row["qwen_answer"] = answer
            row["gemini_judgement"] = verdict
            row["judge_model_name"] = active_judge_model()
            merged[row["question_number"]] = row
            print(row["question_number"], verdict.split("\n")[0])
    BATCH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged.values()), encoding="utf-8")
    print(f"{sum(1 for r in merged.values() if r['gemini_judgement'].startswith(HALL))}/{len(merged)} hallucinating")


def _extract_claim(question: str, answer: str, dry_run: bool) -> tuple[str, list[str]]:
    if dry_run:
        return answer[:200], []
    claim = gpt(P_CLAIM.format(q=question[:1500], a=answer[:3000]))
    text = strip_thinking(str(claim.get("claim", "")))[:800]
    entities = [str(e) for e in claim.get("entities", [])][:4]
    return text, entities


def _judge_turn(question: str, claim: str, answer: str, messages: list[dict], reply: str, dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "drop", "dry-run"
    payload = gpt(P_TURN.format(
        q=question[:1500], claim=claim, a=answer[:2500],
        hist=history(messages), last=reply[:2500],
    ))
    if isinstance(payload, dict):
        label = parse_judge_label(str(payload.get("label", "")))
        reason = str(payload.get("reason", ""))
        return label, reason
    return parse_judge_label(str(payload)), str(payload)[:300]


def _maybe_features(args, question: str, answer: str) -> dict:
    if args.dry_run or os.environ.get("SKIP_FEATURES") == "1":
        return {}
    try:
        import runtime
        from features import calculate_features
        if runtime.model is None:
            return {}
        return {
            f"init_{name}": value
            for name, value in calculate_features(
                runtime.tokenizer, runtime.model, runtime.device, question, answer
            ).items()
        }
    except Exception as error:
        print(f"Warning: seed features skipped ({error})")
        return {}


def cmd_tree(args) -> None:
    """Step C: one pure-strategy branch per category, judged turn-by-turn."""
    cats = list(CATS) if args.categories == "all" else args.categories.split(",")
    if bad := [c for c in cats if c not in CATS]:
        raise SystemExit(f"Unknown categories {bad}; choose from {list(CATS)}")

    raw_seeds = [
        r for r in rows(Path(args.seeds))
        if hallucinating(r) and not r.get("duplicate_answer")
    ]
    if not raw_seeds:
        raise SystemExit(f"No hallucinating rows in {args.seeds}")
    seeds = sample_seeds(raw_seeds, args.max_seeds)
    plan = sampling_plan(raw_seeds, args.max_seeds)
    out = Path(args.out)
    done = {r["branch_id"] for r in rows(out)} if args.resume else set()
    seen = bool(done)
    from runtime import active_judge_model
    judge_name = active_judge_model()
    print(
        f"{len(seeds)} seeds x {len(cats)} categories x {args.levels} levels = "
        f"{len(seeds) * len(cats)} branches, {len(seeds) * len(cats) * args.levels} answers"
    )
    print(f"Answer model: {args.model} (thinking {'on' if ENABLE_THINKING else 'off'})")
    print(f"Judge model: {judge_name if not args.dry_run else 'dry-run'}")
    print(
        "Sampling: "
        + ", ".join(
            f"{domain} {plan[domain]['selected']}/{plan[domain]['available']}"
            for domain in ("research", "legal", "medical", "total")
        )
    )
    chat = (lambda m: f"[stub] {m[-1]['content'][:60]}") if args.dry_run else load_qwen(args.model)

    for index, seed in enumerate(seeds, 1):
        question = seed["question"]
        first = strip_question_prefix(
            question,
            first_present_field(seed, ("qwen_answer", "model_answer", "answer", "response"), "seed answer"),
        )
        todo = [c for c in cats if branch_id(args.model, seed, c) not in done]
        if not todo:
            continue
        text, entities = _extract_claim(question, first, args.dry_run)
        features = _maybe_features(args, question, first)
        print(f"\n[{index}/{len(seeds)}] q{seed['question_number']} ({domain_of(seed)}): {text[:90]}")
        for cat in todo:
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": first}]
            label, record, turns = SEED_LABEL, {}, []
            for level in range(1, args.levels + 1):
                state_key = {
                    "correct": "corrected",
                    "repeat": "persisted",
                    "depend": "new_hallucination",
                    "drop": "not_applicable",
                }.get(label, "persisted")
                ask, why = backup(cat, entities, state_key), "dry-run" if args.dry_run else ""
                if not args.dry_run:
                    for _ in range(2):
                        drafted = str(gpt(P_DRAFT.format(
                            q=question[:1500], claim=text, hist=history(messages),
                            state=label, hint=hint_for(label), cat=cat, rule=CATS[cat][0],
                            intent=FOLLOWUP_TYPE_DESCRIPTIONS[cat],
                        )).get("follow_up", "")).strip()
                        if not (why := check(drafted, cat, entities)):
                            ask = drafted
                            break
                messages.append({"role": "user", "content": ask})
                reply = strip_thinking(chat(messages))
                messages.append({"role": "assistant", "content": reply})
                label, reason = _judge_turn(question, text, first, messages, reply, args.dry_run)
                state = display_state(label)
                record |= {
                    f"follow_up_{level}": ask,
                    f"future_turn_{level}": reply,
                    f"turn_state_{level}": state,
                    f"turn_label_{level}": label.upper(),
                    f"turn_reason_{level}": reason,
                    f"rejected_{level}": why,
                }
                turns.append({"turn": level, "label": label, "followup_type": cat, "state": state})
            derived = derive_branch_outcome(turns)
            bid = branch_id(args.model, seed, cat)
            write(out, {
                "schema_version": SCHEMA_VERSION,
                "branch_id": bid,
                "question_number": seed["question_number"],
                "seed_id": seed_identifier(seed),
                "sample_index": seed.get("sample_index"),
                "domain": domain_of(seed),
                "answer_model": args.model,
                "judge_model_name": judge_name,
                "enable_thinking": ENABLE_THINKING,
                "follow_up_mode": cat,
                "question": question,
                "original_answer": first,
                "false_claim": text,
                "claim": text,
                "entities": entities,
                "levels": args.levels,
                "branch_outcome": derived["branch_outcome"],
                "final_label": derived["final_label"],
                "label_counts": derived["label_counts"],
                "first_depend_turn": derived["first_depend_turn"],
                "first_correct_turn": derived["first_correct_turn"],
                **features,
                **record,
            }, seen)
            seen = True
            done.add(bid)
            print(
                f"  {cat:<20} {[record[f'turn_state_{i}'] for i in range(1, args.levels + 1)]} "
                f"-> {derived['branch_outcome']}"
            )
            if not args.dry_run:
                write(LABELS, {
                    "schema_version": SCHEMA_VERSION,
                    "branch_id": bid,
                    "question_number": seed["question_number"],
                    "domain": domain_of(seed),
                    "answer_model": args.model,
                    "follow_up_mode": cat,
                    "reason": "derived from per-turn DROP/CORRECT/REPEAT/DEPEND labels",
                    "final_label": derived["final_label"],
                    "derived_label": derived["branch_outcome"],
                    "label_counts": derived["label_counts"],
                }, Path(LABELS).exists() or seen)
    print(f"\n-> {out}")


def cmd_label(args) -> None:
    """Step D: derive DROP/CORRECT/REPEAT/DEPEND from per-turn labels, with optional LLM check."""
    done = {r["branch_id"] for r in rows(LABELS)} if args.resume else set()
    todo = [r for r in rows(Path(args.tree)) if r["branch_id"] not in done]
    print(f"Labeling {len(todo)} branches")
    seen = bool(done)
    for i, row in enumerate(todo, 1):
        turns = []
        for n in range(1, row.get("levels", 5) + 1):
            if f"future_turn_{n}" not in row:
                continue
            label = parse_judge_label(str(row.get(f"turn_label_{n}", row.get(f"turn_state_{n}", ""))))
            turns.append({"turn": n, "label": label})
        derived = derive_branch_outcome(turns)
        outcome = derived["branch_outcome"]
        reason = "derived from per-turn DROP/CORRECT/REPEAT/DEPEND labels"
        if args.llm_label:
            blob = "\n\n".join(
                f"USER: {row.get(f'follow_up_{n}', '')}\nASSISTANT: {row.get(f'future_turn_{n}', '')[:1500]}"
                for n in range(1, row.get("levels", 5) + 1) if f"future_turn_{n}" in row
            )
            out = gpt(P_LABEL.format(
                q=row["question"][:1500],
                claim=str(row.get("false_claim") or row.get("claim", ""))[:800],
                a=row["original_answer"][:2500],
                turns=blob[:9000],
            ))
            outcome = normalize_outcome(str(out.get("final_label", "")))
            reason = str(out.get("reason", reason))
        write(LABELS, {
            "schema_version": SCHEMA_VERSION,
            "branch_id": row["branch_id"],
            "question_number": row["question_number"],
            "domain": row.get("domain") or domain_of(row),
            "answer_model": row.get("answer_model"),
            "follow_up_mode": row["follow_up_mode"],
            "reason": reason,
            "final_label": outcome,
            "derived_label": derived["branch_outcome"],
            "label_counts": derived["label_counts"],
        }, seen)
        seen = True
        print(f"[{i}/{len(todo)}] {row['branch_id']} -> {outcome}")
    cmd_report(args)


def cmd_seeds(_args) -> None:
    """Generate per-model hallucinating seeds (HallucinationResearchTest sampler)."""
    import generate_seeds
    generate_seeds.main()


def cmd_report(args) -> None:
    from report import render_report
    render_report(
        from_partial=getattr(args, "from_partial", False),
        tree_path=Path(getattr(args, "tree", TREE)),
        labels_path=LABELS,
        html_path=Path(getattr(args, "html", DIR / "results" / "cascade_report.html")),
        pdf_path=Path(getattr(args, "pdf", DIR / "results" / "cascade_report.pdf")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    answer = sub.add_parser("answer", help="generate turn-0 answers")
    answer.add_argument("--n", type=int, default=None, help="questions per domain")
    judge = sub.add_parser("judge", help="mark answers hallucinating or not")
    tree = sub.add_parser("tree", help="build the follow-up tree")
    tree.add_argument("--seeds", default=str(BATCH), help="any JSONL with question + answer + judgement")
    tree.add_argument("--out", default=str(TREE))
    tree.add_argument("--max-seeds", type=int, default=DEFAULT_MAX_SEEDS)
    tree.add_argument("--levels", type=int, default=DEFAULT_TURNS)
    tree.add_argument("--categories", default="all")
    tree.add_argument("--dry-run", action="store_true", help="stub answers, no GPU or API")
    label = sub.add_parser("label", help="label branch outcomes")
    label.add_argument("--tree", default=str(TREE))
    label.add_argument("--llm-label", action="store_true", help="ask the judge model instead of deriving")
    sub.add_parser("seeds", help="generate per-model hallucinating seeds")
    report = sub.add_parser("report", help="outcome tables, CIs, and HTML/PDF")
    report.add_argument("--from-partial", action="store_true", help="use the formatted 61-seed captured run")
    report.add_argument("--tree", default=str(TREE))
    report.add_argument("--html", default=str(DIR / "results" / "cascade_report.html"))
    report.add_argument("--pdf", default=str(DIR / "results" / "cascade_report.pdf"))
    for name in ("answer", "judge"):
        sub.choices[name].add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
    for name in ("answer", "judge", "tree", "label"):
        sub.choices[name].add_argument("--resume", action="store_true")
    for name in ("answer", "tree"):
        sub.choices[name].add_argument("--model", default=QWEN)
    args = parser.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


# Re-export contract helpers so existing tests keep `from pipeline import ...`.
__all__ = ["CATS", "STATES", "backup", "check", "names", "main"]


if __name__ == "__main__":
    main()
