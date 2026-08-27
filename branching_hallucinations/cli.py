"""Stage-oriented CLI for the Branching Hallucinations experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys

from .analysis import analyze
from .config import DEFAULT_CONFIG, load_config
from .interventions import audit_action
from .models import ExperimentSamplers
from .schemas import (
    ParseStatus,
    VerificationStatus,
    VerifiedSeed,
)
from .seeds import extract_claims_for_seed, generate_seeds
from .storage import (
    RunStore,
    conversation_before_user_turn,
    conversation_for,
    render_conversation,
    write_json,
)
from .trajectory_judge import judge_trajectory
from .tree import expected_node_count, generate_tree


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML config path")
    common.add_argument("--run", required=False, help="Run directory, e.g. runs/pilot")

    parser = argparse.ArgumentParser(
        prog="branching_hallucinations",
        description="Stage-separated Branching Hallucinations experiment.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-run", parents=[common], help="Create run directory and write manifest")

    gen = sub.add_parser("generate-seeds", parents=[common], help="Generate GPT-OSS seed answers")
    gen.add_argument("--n", type=int, default=None)

    sub.add_parser("extract-claims", parents=[common], help="Extract atomic candidate claims")

    ver = sub.add_parser("verify-seeds", parents=[common], help="Ground candidates and freeze VERIFIED_FALSE seeds")
    ver.add_argument("--max-verified", type=int, default=None)

    tree = sub.add_parser("generate-tree", parents=[common], help="Generate D/N/V tree without trajectory labels")
    tree.add_argument("--max-seeds", type=int, default=None)

    audit = sub.add_parser("audit-actions", parents=[common], help="Post-hoc D/N/V action audit")
    audit.add_argument("--version", default="v1")

    judge = sub.add_parser("judge-trajectories", parents=[common], help="Judge frozen tree nodes")
    judge.add_argument("--version", default="v1")

    an = sub.add_parser("analyze", parents=[common], help="Paired/clustered analysis of frozen artifacts")
    an.add_argument("--trajectory-version", default="v1")
    an.add_argument("--audit-version", default="v1")

    exp = sub.add_parser("export-audit", parents=[common], help="Export compact human-audit CSVs")
    exp.add_argument("--trajectory-version", default="v1")
    exp.add_argument("--audit-version", default="v1")
    return parser


def _store(args) -> RunStore:
    if not args.run:
        raise SystemExit("--run is required")
    store = RunStore(args.run)
    store.ensure()
    return store


async def cmd_init_run(args) -> None:
    config = load_config(args.config)
    store = _store(args)
    store.write_manifest(config)
    print(f"Initialized {store.root} schema={config.schema_version}")


async def cmd_generate_seeds(args) -> None:
    config = load_config(args.config)
    store = _store(args)
    store.write_manifest(config)
    samplers = ExperimentSamplers.from_config(config)
    n_seeds = args.n or config.n_seeds
    created = await generate_seeds(
        store,
        answer_model=samplers.answer,
        domains=config.domains,
        n_seeds=n_seeds,
        max_questions=config.max_questions,
        samples_per_question=config.samples_per_question,
        answer_name=config.answer.sampler,
        dataset=config.dataset,
    )
    print(f"Wrote {len(created)} new seeds to {store.generated_seeds_path}")


async def cmd_extract_claims(args) -> None:
    config = load_config(args.config)
    store = _store(args)
    samplers = ExperimentSamplers.from_config(config)
    done_seeds = {claim.seed_id for claim in store.candidate_claims()}
    n_new = 0
    for seed in store.generated_seeds():
        if seed.seed_id in done_seeds:
            continue
        claims = await extract_claims_for_seed(
            seed, samplers.claim_extractor, max_claims=config.max_claims_per_seed
        )
        if not claims:
            print(f"{seed.seed_id}: no parseable claims (excluded, not labeled false)")
            continue
        for claim in claims:
            store.append_claim(claim)
            n_new += 1
        print(f"{seed.seed_id}: {len(claims)} candidate claims")
    print(f"Wrote {n_new} claims to {store.candidate_claims_path}")


async def cmd_verify_seeds(args) -> None:
    from .grounding import verify_claim

    config = load_config(args.config)
    store = _store(args)
    samplers = ExperimentSamplers.from_config(config)
    seeds = store.generated_by_id()
    claims = store.candidate_claims()
    done = store.completed_ids(store.verification_path, "claim_id")
    already_verified = {item.seed_id for item in store.verified_seeds()}
    max_verified = args.max_verified or config.n_seeds
    n_verified = len(already_verified)
    by_seed: dict[str, list] = {}
    for claim in claims:
        by_seed.setdefault(claim.seed_id, []).append(claim)
    for seed_id, seed_claims in by_seed.items():
        if n_verified >= max_verified:
            break
        if seed_id in already_verified:
            continue
        seed = seeds.get(seed_id)
        if seed is None:
            continue
        chosen = None
        for claim in seed_claims:
            if claim.claim_id in done:
                continue
            result = await verify_claim(
                claim=claim.text,
                context=seed.seed_answer,
                domain=seed.domain,
                search_sampler=samplers.search_planner,
                verifier=samplers.grounded_verifier,
                method=config.grounding_method,
                max_searches=config.max_searches,
                claim_id=claim.claim_id,
                grounding_task=config.dataset.task_for(seed.domain),
            )
            store.append_verification(result)
            done.add(claim.claim_id)
            print(f"{claim.claim_id}: {result.status.value}")
            if result.status is VerificationStatus.VERIFIED_FALSE and result.parse_status.value != "failed":
                chosen = (claim, result)
                break
        if chosen is None:
            continue
        claim, result = chosen
        verified = VerifiedSeed(
            seed_id=seed.seed_id,
            question_id=seed.question_id,
            domain=seed.domain,
            question=seed.question,
            seed_answer=seed.seed_answer,
            tracked_claim=claim.text,
            tracked_claim_id=claim.claim_id,
            verification_status=VerificationStatus.VERIFIED_FALSE,
            verification_reason=result.reason,
            queries=result.queries,
            sources=result.sources,
            evidence_passages=result.evidence_passages,
            answer_model_metadata=seed.generation_metadata,
            verification_metadata=result.to_dict(),
        )
        store.append_verified(verified)
        n_verified += 1
        print(f"FROZEN {verified.seed_id}: {verified.tracked_claim[:120]}")
    print(f"Verified-false seeds: {len(store.verified_seeds())} in {store.verified_seeds_path}")


async def cmd_generate_tree(args) -> None:
    config = load_config(args.config)
    store = _store(args)
    samplers = ExperimentSamplers.from_config(config)
    seeds = store.verified_seeds()
    max_seeds = args.max_seeds or config.n_seeds
    seeds = seeds[:max_seeds]
    if not seeds:
        raise SystemExit("No verified-false seeds. Run verify-seeds first.")
    print(
        f"{len(seeds)} verified-false seeds, depth={config.depth}, "
        f"{expected_node_count(len(config.actions), config.depth)} nodes/seed. "
        "Trajectory judge is not called."
    )
    created = await generate_tree(
        store,
        seeds,
        answer_model=samplers.answer,
        writer=samplers.followup_writer,
        actions=config.actions,
        depth=config.depth,
    )
    print(f"Wrote {len(created)} new nodes to {store.nodes_path}")


async def cmd_audit_actions(args) -> None:
    config = load_config(args.config)
    store = _store(args)
    samplers = ExperimentSamplers.from_config(config)
    seeds = store.seeds_by_id()
    nodes_by_id = store.nodes_by_id()
    done = store.completed_ids(store.action_audit_path(args.version), "node_id")
    n = 0
    for node in store.nodes():
        if node.node_id in done:
            continue
        seed = seeds[node.seed_id]
        before = conversation_before_user_turn(node, seed, nodes_by_id)
        audit = await audit_action(
            tracked_claim=seed.tracked_claim,
            conversation_before_user_message=before,
            user_message=node.user_message,
            desired_action=node.action,
            auditor=samplers.followup_writer,
        )
        audit.node_id = node.node_id
        audit.auditor_metadata = {
            **audit.auditor_metadata,
            "fallback_used": bool((node.intervention_metadata or {}).get("fallback_used")),
        }
        store.append_action_audit(audit, version=args.version)
        n += 1
    print(f"Wrote {n} action audits to {store.action_audit_path(args.version)}")


async def cmd_judge_trajectories(args) -> None:
    config = load_config(args.config)
    store = _store(args)
    samplers = ExperimentSamplers.from_config(config)
    seeds = store.seeds_by_id()
    nodes_by_id = store.nodes_by_id()
    done = store.completed_ids(store.trajectory_path(args.version), "node_id")
    n = 0
    for node in store.nodes():
        if node.node_id in done:
            continue
        seed = seeds[node.seed_id]
        convo = conversation_for(node.node_id, seed, nodes_by_id)
        judgment = await judge_trajectory(
            node_id=node.node_id,
            seed_id=node.seed_id,
            tracked_claim=seed.tracked_claim,
            seed_answer=seed.seed_answer,
            conversation=convo,
            latest_response=node.assistant_response,
            judge=samplers.trajectory_judge,
        )
        store.append_judgment(judgment, version=args.version)
        n += 1
        status = judgment.parse_status.value
        label = "UNPARSED" if judgment.parse_status is ParseStatus.FAILED else judgment.label.value
        print(f"{node.node_id}: {status} {label}")
    print(f"Wrote {n} judgments to {store.trajectory_path(args.version)}")


def cmd_analyze(args) -> None:
    store = _store(args)
    summary = analyze(
        verified_seeds=store.verified_seeds(),
        nodes=store.nodes(),
        judgments=store.judgments(args.trajectory_version),
        audits=store.action_audits(args.audit_version),
        verifications=store.verifications(),
        out_dir=store.analysis_dir,
        random_seed=store.load_manifest().get("random_seed", 42),
    )
    write_json(store.reports_dir / "summary.json", summary)
    print(f"Analysis written to {store.analysis_dir}")
    print(
        f"Seeds={summary['n_verified_seeds']} nodes={summary['n_nodes']} "
        f"failed_judgments={summary['n_judgments_failed']}"
    )


def cmd_export_audit(args) -> None:
    store = _store(args)
    seeds = store.seeds_by_id()
    nodes_by_id = store.nodes_by_id()
    audits = {item.node_id: item for item in store.action_audits(args.audit_version)}
    judgments = {item.node_id: item for item in store.judgments(args.trajectory_version)}
    seed_path = store.reports_dir / "seed_audit.csv"
    intervention_path = store.reports_dir / "intervention_audit.csv"
    trajectory_path = store.reports_dir / "trajectory_audit.csv"
    store.reports_dir.mkdir(parents=True, exist_ok=True)
    with seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_id",
                "question",
                "seed_answer",
                "tracked_claim",
                "verification_status",
                "verification_reason",
                "queries",
                "sources",
            ],
        )
        writer.writeheader()
        for seed in store.verified_seeds():
            writer.writerow(
                {
                    "seed_id": seed.seed_id,
                    "question": seed.question,
                    "seed_answer": seed.seed_answer,
                    "tracked_claim": seed.tracked_claim,
                    "verification_status": seed.verification_status.value,
                    "verification_reason": seed.verification_reason,
                    "queries": " | ".join(seed.queries),
                    "sources": " | ".join(seed.sources),
                }
            )
    with intervention_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_id",
                "path",
                "tracked_claim",
                "conversation_before",
                "desired_action",
                "user_message",
                "realized_action",
                "valid",
                "fallback_used",
            ],
        )
        writer.writeheader()
        for node in store.nodes():
            seed = seeds[node.seed_id]
            before = conversation_before_user_turn(node, seed, nodes_by_id)
            audit = audits.get(node.node_id)
            writer.writerow(
                {
                    "seed_id": node.seed_id,
                    "path": "/".join(node.path),
                    "tracked_claim": seed.tracked_claim,
                    "conversation_before": render_conversation(before),
                    "desired_action": node.action.value,
                    "user_message": node.user_message,
                    "realized_action": audit.realized_action.value if audit and audit.realized_action else "",
                    "valid": audit.valid if audit else "",
                    "fallback_used": (node.intervention_metadata or {}).get("fallback_used"),
                }
            )
    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_id",
                "path",
                "tracked_claim",
                "conversation",
                "latest_response",
                "model_label",
                "parse_status",
                "evidence_spans",
                "human_label",
            ],
        )
        writer.writeheader()
        for node in store.nodes():
            seed = seeds[node.seed_id]
            convo = conversation_for(node.node_id, seed, nodes_by_id)
            judgment = judgments.get(node.node_id)
            writer.writerow(
                {
                    "seed_id": node.seed_id,
                    "path": "/".join(node.path),
                    "tracked_claim": seed.tracked_claim,
                    "conversation": render_conversation(convo[:-1] if convo else []),
                    "latest_response": node.assistant_response,
                    "model_label": judgment.label.value if judgment else "",
                    "parse_status": judgment.parse_status.value if judgment else "",
                    "evidence_spans": " | ".join(span.text for span in (judgment.evidence_spans if judgment else [])),
                    "human_label": "",
                }
            )
    print(f"Wrote {seed_path}")
    print(f"Wrote {intervention_path}")
    print(f"Wrote {trajectory_path}")


COMMANDS = {
    "init-run": cmd_init_run,
    "generate-seeds": cmd_generate_seeds,
    "extract-claims": cmd_extract_claims,
    "verify-seeds": cmd_verify_seeds,
    "generate-tree": cmd_generate_tree,
    "audit-actions": cmd_audit_actions,
    "judge-trajectories": cmd_judge_trajectories,
    "analyze": cmd_analyze,
    "export-audit": cmd_export_audit,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = COMMANDS[args.command]
    if args.command in {"analyze", "export-audit"}:
        command(args)
        return 0
    asyncio.run(command(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
