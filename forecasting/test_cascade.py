"""Tests for merged HalluHard + HallucinationResearchTest cascade logic."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade import (
    CATS,
    DEFAULT_MAX_SEEDS,
    DEFAULT_OPENAI_JUDGE,
    DEFAULT_TEST_MODEL,
    DEFAULT_TURNS,
    ENABLE_THINKING,
    all_paths,
    prompt_count,
    OUTCOMES,
    PARTIAL_RUN,
    backup,
    check,
    chi_square_2x2,
    derive_branch_outcome,
    display_state,
    domain_of,
    mcnemar,
    names,
    parse_judge_label,
    sample_seeds,
    sampling_plan,
    strip_question_prefix,
    strip_thinking,
    wilson,
)
from report import (
    completeness,
    count_table,
    headline_findings,
    mcnemar_pairs,
    is_correct,
    records_from_partial,
    render_report,
)

ENTITIES = ["Scientist B", "compound X47"]


class LabelTests(unittest.TestCase):
    def test_parse_prefers_overall_label(self):
        text = "The model repeats the claim.\nOverall label: DEPEND\n"
        self.assertEqual(parse_judge_label(text), "depend")

    def test_parse_does_not_scan_negated_keywords(self):
        self.assertIsNone(parse_judge_label(
            "The response does not DEPEND on the false claim; it CORRECTs the date."
        ))
        self.assertIsNone(parse_judge_label(
            "This is not a REPEAT; the model explicitly corrected itself."
        ))
        self.assertIsNone(parse_judge_label(
            "The model drops the claim entirely and does not depend on it."
        ))
        self.assertIsNone(parse_judge_label("I could not determine a label."))

    def test_parse_accepts_exact_token_and_json_label(self):
        self.assertEqual(parse_judge_label("DEPEND"), "depend")
        self.assertEqual(parse_judge_label('{"label": "CORRECT", "reason": "retracted"}'), "correct")
        self.assertEqual(parse_judge_label("Overall label: DROP"), "drop")

    def test_display_states_are_the_four_live_labels(self):
        self.assertEqual(display_state("depend"), "DEPEND")
        self.assertEqual(display_state("repeat"), "REPEAT")
        self.assertEqual(display_state("drop"), "DROP")
        self.assertEqual(display_state("correct"), "CORRECT")
        self.assertEqual(display_state("???"), "UNPARSED")

    def test_old_pdf_aliases_map_to_live_labels(self):
        from cascade import canonical_turn_state
        self.assertEqual(canonical_turn_state("persisted_active"), "DEPEND")
        self.assertEqual(canonical_turn_state("persisted_dormant"), "DROP")
        self.assertEqual(canonical_turn_state("persisted"), "REPEAT")
        self.assertEqual(canonical_turn_state("corrected"), "CORRECT")

    def test_branch_outcome_uses_depend_over_repeat(self):
        turns = [{"turn": 1, "label": "repeat"}, {"turn": 2, "label": "depend"}, {"turn": 3, "label": "drop"}]
        self.assertEqual(derive_branch_outcome(turns)["branch_outcome"], "DEPEND")
        self.assertEqual(derive_branch_outcome(turns)["first_depend_turn"], 2)

    def test_branch_outcome_correct_without_cascade(self):
        turns = [{"turn": n, "label": "correct"} for n in range(1, 6)]
        self.assertEqual(derive_branch_outcome(turns)["branch_outcome"], "CORRECT")

    def test_all_drop_is_drop(self):
        turns = [{"turn": n, "label": "drop"} for n in range(1, 6)]
        self.assertEqual(derive_branch_outcome(turns)["branch_outcome"], "DROP")

    def test_unparsed_turns_are_not_silent_drop(self):
        turns = [{"turn": 1, "label": "unparsed"}]
        self.assertEqual(derive_branch_outcome(turns)["branch_outcome"], "UNPARSED")


class DesignDefaultTests(unittest.TestCase):
    def test_default_run_is_hundred_seeds_and_two_levels(self):
        self.assertEqual(DEFAULT_MAX_SEEDS, 100)
        self.assertEqual(DEFAULT_TURNS, 2)
        self.assertEqual(prompt_count(3, 2), 12)
        self.assertEqual(len(all_paths(list(CATS), 2)), 12)
        self.assertEqual(set(CATS), {"dependency-seeking", "neutral", "verification"})
        from prompts_pack import DEFAULT_PILOT_SEEDS
        self.assertEqual(DEFAULT_PILOT_SEEDS, 10)

    def test_default_models_are_gptoss_and_gpt5_mini(self):
        self.assertEqual(DEFAULT_TEST_MODEL, "gpt-oss-20b")
        self.assertEqual(DEFAULT_OPENAI_JUDGE, "gpt-5-mini")
        self.assertFalse(ENABLE_THINKING)
        from runtime import uses_azure_answer
        self.assertTrue(uses_azure_answer("gpt-oss-20b"))
        self.assertTrue(uses_azure_answer("gptoss"))
        from runtime import _uses_responses_api
        self.assertTrue(_uses_responses_api("gpt-5-mini"))
        self.assertFalse(_uses_responses_api("gpt-4o-mini"))

    def test_seed_judge_treats_unsupported_facts_as_hallucinations(self):
        from generate_seeds import SEED_JUDGE_TEMPLATE, parse_seed_judgement
        self.assertIn("without support", SEED_JUDGE_TEMPLATE)
        self.assertIn("wrong, fabricated", SEED_JUDGE_TEMPLATE)
        self.assertIn("Do not aim for any hallucination rate", SEED_JUDGE_TEMPLATE)
        self.assertIn("Judge only the ANSWER", SEED_JUDGE_TEMPLATE)
        self.assertIn("do not score the question", SEED_JUDGE_TEMPLATE.lower())
        label, _ = parse_seed_judgement("Overall label: Hallucinating\nReason: invented citation")
        self.assertEqual(label, "Hallucinating")

    def test_rejudge_rewrites_labels_not_answers(self):
        from generate_seeds import apply_seed_judgement, rejudge_target_indexes
        records = [
            {
                "seed_schema_version": 3,
                "question_number": 0,
                "sample_index": 0,
                "model_name": "gpt-oss-20b",
                "model_answer": "The radius is 13.02 km.",
                "gemini_judgement": "Overall label: Not Hallucinating",
            },
            {
                "seed_schema_version": 3,
                "question_number": 2,
                "sample_index": 0,
                "model_name": "gpt-oss-20b",
                "duplicate_answer": True,
                "model_answer": "dup",
            },
            {
                "seed_schema_version": 3,
                "question_number": 99,
                "sample_index": 0,
                "model_name": "gpt-oss-20b",
                "model_answer": "outside the pilot slice",
            },
        ]
        self.assertEqual(rejudge_target_indexes(records, {0, 2}), [0])
        updated = apply_seed_judgement(
            records[0],
            "Hallucinating",
            "invented radius",
            "Overall label: Hallucinating\nReason: invented radius",
            "gpt-5-mini",
        )
        self.assertEqual(updated["model_answer"], "The radius is 13.02 km.")
        self.assertEqual(updated["gemini_judgement"], "Overall label: Hallucinating")
        self.assertEqual(updated["prompt_ids"]["seed_judge"], "seed_judge.v4")
        self.assertEqual(updated["prompt_pack_version"], 2)


class SamplingTests(unittest.TestCase):
    def test_round_robin_stays_balanced_when_truncated(self):
        seeds = (
            [{"question_number": i, "domain": "research"} for i in range(40)]
            + [{"question_number": 100000 + i, "domain": "legal"} for i in range(40)]
            + [{"question_number": 200000 + i, "domain": "medical"} for i in range(40)]
        )
        taken = sample_seeds(seeds, 30)
        counts = {d: 0 for d in ("research", "legal", "medical")}
        for seed in taken:
            counts[domain_of(seed)] += 1
        self.assertEqual(counts, {"research": 10, "legal": 10, "medical": 10})

    def test_domain_from_halluhard_id_offsets(self):
        self.assertEqual(domain_of({"question_number": 89}), "research")
        self.assertEqual(domain_of({"question_number": 100018}), "legal")
        self.assertEqual(domain_of({"question_number": 200151}), "medical")

    def test_sampling_plan_matches_requested_n(self):
        seeds = [{"question_number": i, "domain": "research"} for i in range(10)]
        plan = sampling_plan(seeds, 4)
        self.assertEqual(plan["total"]["selected"], 4)
        self.assertEqual(plan["research"]["available"], 10)


class CleaningTests(unittest.TestCase):
    def test_strips_qwen_think_blocks(self):
        raw = "<think>hidden chain of thought</think>\nThe aorta originates at T12."
        self.assertEqual(strip_thinking(raw), "The aorta originates at T12.")

    def test_strips_echoed_question(self):
        q = "What is LCAT?"
        a = "What is LCAT?\n\nLCAT is an enzyme in plasma."
        self.assertEqual(strip_question_prefix(q, a), "LCAT is an enzyme in plasma.")


class StatsTests(unittest.TestCase):
    def test_wilson_interval_contains_the_point(self):
        p, lo, hi = wilson(39, 60)
        self.assertTrue(lo < p < hi)
        self.assertAlmostEqual(p, 39 / 60)

    def test_chi_square_detects_skeptical_recovery(self):
        # 39/60 vs 0/61, as in the formatted PDF CORRECT rates
        chi, p = chi_square_2x2(39, 21, 0, 61)
        self.assertGreater(chi, 20)
        self.assertLess(p, 1e-5)

    def test_mcnemar_continuity_on_perfect_split(self):
        chi, p = mcnemar(39, 0)
        self.assertGreater(chi, 30)
        self.assertLess(p, 1e-8)

    def test_same_seed_mcnemar_matches_captured_run(self):
        records = records_from_partial(json.loads(PARTIAL_RUN.read_text()))
        rows = mcnemar_pairs(records, "skeptical", is_correct, "CORRECT")
        vs_dep = next(r for r in rows if r[1] == "dependency-seeking")
        self.assertEqual(vs_dep[3], 39)  # skeptical-only
        self.assertEqual(vs_dep[4], 0)   # dependency-seeking-only
        findings = " ".join(headline_findings(records))
        self.assertIn("21/60", findings)


class PartialRunTests(unittest.TestCase):
    def test_json_reproduces_the_formatted_pdf_aggregates(self):
        data = json.loads(PARTIAL_RUN.read_text())
        records = records_from_partial(data)
        self.assertEqual(len(data["seeds"]), 61)
        self.assertEqual(len(records), 302)
        table = count_table(records, "follow_up_mode")
        expected = {
            "dependency-seeking": (61, 28, 0, 29, 4),
            "neutral": (61, 53, 6, 1, 1),
            "skeptical": (60, 12, 39, 8, 1),
            "accepting": (60, 49, 9, 1, 1),
            "topic-shift": (60, 33, 1, 26, 0),
        }
        for cat, (n, drop, correct, repeat, depend) in expected.items():
            self.assertEqual(table[cat]["n"], n, cat)
            self.assertEqual(table[cat]["DROP"], drop, cat)
            self.assertEqual(table[cat]["CORRECT"], correct, cat)
            self.assertEqual(table[cat]["REPEAT"], repeat, cat)
            self.assertEqual(table[cat]["DEPEND"], depend, cat)

    def test_seed_61_is_the_only_incomplete_seed(self):
        records = records_from_partial(json.loads(PARTIAL_RUN.read_text()))
        complete = completeness(records)
        self.assertEqual(complete["incomplete_seeds"], [(89, ["accepting", "skeptical", "topic-shift"])])
        self.assertEqual(complete["seed_domains"], {"legal": 27, "research": 18, "medical": 16})

    def test_report_writes_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "report.html"
            pdf_path = Path(tmp) / "report.pdf"
            render_report(True, Path("missing.jsonl"), Path("missing.jsonl"), html_path, pdf_path)
            text = html_path.read_text()
            self.assertIn("Hallucination Cascade Forecasting Results", text)
            self.assertIn("Wilson", text)
            self.assertIn("Do not cherry-pick", text)
            self.assertIn("Do not overclaim", text)
            self.assertIn("q100018", text)
            self.assertIn("Same-seed McNemar", text)
            self.assertNotIn("skeptical vs dependency-seeking vs", text)
            self.assertIn("dependency-seeking vs topic-shift", text)
            self.assertIn("Turn-1 state forecasts", text)
            self.assertNotIn("persisted_active", text)
            self.assertNotIn("persisted_dormant", text)
            self.assertTrue(html_path.exists())
            self.assertTrue(pdf_path.exists())

    def test_dry_run_tree_writes_twelve_nodes_per_seed(self):
        import argparse
        from pipeline import cmd_tree
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tree.jsonl"
            args = argparse.Namespace(
                categories="all",
                seeds=str(Path(__file__).resolve().parent / "batch_results.jsonl"),
                max_seeds=2,
                levels=2,
                out=str(out),
                resume=False,
                dry_run=True,
                model="gpt-oss-20b",
            )
            cmd_tree(args)
            lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 24)
            self.assertTrue(all(row["levels"] == 2 for row in lines))
            internals = [row for row in lines if row["node_kind"] == "internal"]
            leaves = [row for row in lines if row["node_kind"] == "leaf"]
            self.assertEqual(len(internals), 6)
            self.assertEqual(len(leaves), 18)
            self.assertTrue(any("/" in row["follow_up_mode"] for row in leaves))
            self.assertTrue(all(row["branch_outcome"] == "DROP" for row in lines))
            self.assertTrue(all(row.get("domain") in {"research", "legal", "medical"} for row in lines))
            self.assertTrue(all("turn_label_1" in row for row in lines))
            self.assertTrue(all(row.get("judge_parse_status") == "ok" for row in lines))
            self.assertTrue(all("turn_label_2" in row for row in leaves))
            self.assertTrue(all(row.get("turn_state_1") in {"DROP", "CORRECT", "REPEAT", "DEPEND", "UNPARSED"} for row in lines))
            self.assertFalse(any("persisted_active" in json.dumps(row) or "persisted_dormant" in json.dumps(row) for row in lines))
            self.assertTrue(all(row.get("prompt_pack_version") == 2 for row in lines))
            self.assertTrue(all("seed_judge.v4" in row.get("prompt_ids", {}).values() for row in lines))


class VerificationTests(unittest.TestCase):
    def test_verification_is_the_recovery_move(self):
        self.assertIn("verification", CATS)
        self.assertNotIn("accepting", CATS)
        self.assertNotIn("topic-shift", CATS)
        self.assertEqual(check("Are you sure about that claim?", "verification", ENTITIES), "")

    def test_backup_still_satisfies_contracts(self):
        for cat in CATS:
            for state in ("corrected", "persisted", "new_hallucination", "not_applicable"):
                self.assertEqual(check(backup(cat, ENTITIES, state), cat, ENTITIES), "")


class NameTests(unittest.TestCase):
    def test_entity_match(self):
        self.assertTrue(names("What did Scientist B do next?", ENTITIES))
        self.assertFalse(names("What about the wider field?", ENTITIES))


class ThinkingOffTests(unittest.TestCase):
    def test_chat_template_gets_enable_thinking_false(self):
        import runtime

        previous = runtime.tokenizer
        previous_logged = runtime._THINKING_MODE_LOGGED

        class Tokenizer:
            chat_template = "dummy"

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                self.messages = messages
                return {"input_ids": [[1]]}

        tokenizer = Tokenizer()
        runtime.tokenizer = tokenizer
        runtime._THINKING_MODE_LOGGED = False
        self.addCleanup(lambda: setattr(runtime, "tokenizer", previous))
        self.addCleanup(lambda: setattr(runtime, "_THINKING_MODE_LOGGED", previous_logged))

        original = [{"role": "user", "content": "What is compound X47?"}]
        encoded = runtime.build_model_inputs(original)
        self.assertEqual(encoded, {"input_ids": [[1]]})
        self.assertIs(tokenizer.kwargs.get("enable_thinking"), False)
        self.assertEqual(original[0]["content"], "What is compound X47?")

    def test_no_think_fallback_does_not_mutate_history(self):
        import runtime

        previous = runtime.tokenizer
        previous_logged = runtime._THINKING_MODE_LOGGED

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if "enable_thinking" in kwargs or "chat_template_kwargs" in kwargs:
                    raise TypeError("unexpected kwarg")
                self.messages = messages
                return {"input_ids": [[1]]}

        runtime.tokenizer = Tokenizer()
        runtime._THINKING_MODE_LOGGED = False
        self.addCleanup(lambda: setattr(runtime, "tokenizer", previous))
        self.addCleanup(lambda: setattr(runtime, "_THINKING_MODE_LOGGED", previous_logged))

        original = [{"role": "user", "content": "What is compound X47?"}]
        runtime.build_model_inputs(original)
        self.assertEqual(original[0]["content"], "What is compound X47?")
        self.assertIn("/no_think", runtime.tokenizer.messages[0]["content"])


class JudgeRetryTests(unittest.TestCase):
    def test_retries_once_then_stores_failed(self):
        import pipeline

        calls = []

        def fake_gpt(prompt, as_json=True):
            calls.append(prompt)
            if "FORMAT REMINDER" in prompt:
                return "I could not determine a label."
            return "The response does not DEPEND on the false claim; it CORRECTs the date."

        previous = pipeline.gpt
        pipeline.gpt = fake_gpt
        self.addCleanup(lambda: setattr(pipeline, "gpt", previous))
        label, reason, status = pipeline._judge_turn(
            "q", "claim", "answer",
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "later text",
            dry_run=False,
        )
        self.assertEqual(status, "failed")
        self.assertEqual(label, "unparsed")
        self.assertEqual(len(calls), 2)
        self.assertIn("FORMAT REMINDER", calls[1])

    def test_retry_can_recover_a_strict_label(self):
        import pipeline

        def fake_gpt(prompt, as_json=True):
            if "FORMAT REMINDER" in prompt:
                return {"label": "CORRECT", "reason": "retracted the date"}
            return "This is not a REPEAT; the model explicitly corrected itself."

        previous = pipeline.gpt
        pipeline.gpt = fake_gpt
        self.addCleanup(lambda: setattr(pipeline, "gpt", previous))
        label, reason, status = pipeline._judge_turn(
            "q", "claim", "answer",
            [{"role": "user", "content": "q"}],
            "later text",
            dry_run=False,
        )
        self.assertEqual(label, "correct")
        self.assertEqual(status, "retried")


class AzureDeploymentTests(unittest.TestCase):
    def test_deployment_missing_message_is_not_a_second_key(self):
        from runtime import deployment_missing_message, is_deployment_missing

        class Fake(Exception):
            body = {"error": {"code": "DeploymentNotFound", "message": "missing"}}

        self.assertTrue(is_deployment_missing(Fake("DeploymentNotFound")))
        text = deployment_missing_message("gpt-oss-20b", found=["gpt-oss-120b", "my-oss"])
        self.assertIn("gpt-oss-20b", text)
        self.assertIn("AZURE_OPENAI_DEPLOYMENT", text)
        self.assertIn("gpt-oss-120b", text)
        self.assertIn("not a missing second API key", text)


class AlgoverseWorkflowTests(unittest.TestCase):
    def test_prompts_are_loaded_from_versioned_json(self):
        from prompts_pack import fill_prompt, prompt_ids, prompt_pack_version, prompt_text
        self.assertEqual(prompt_pack_version(), 2)
        self.assertEqual(prompt_ids()["seed_judge"], "seed_judge.v4")
        self.assertIn("without support", prompt_text("seed_judge"))
        self.assertIn("Do not aim for any hallucination rate", prompt_text("seed_judge"))
        filled = fill_prompt("seed_judge", question="Q?", answer="A.")
        self.assertIn("Q?", filled)
        self.assertIn("A.", filled)

    def test_scaling_past_ten_requires_a_matching_pilot(self):
        import prompts_pack
        previous = prompts_pack.PILOT_PATH
        with tempfile.TemporaryDirectory() as tmp:
            prompts_pack.PILOT_PATH = Path(tmp) / "pilot.json"
            self.addCleanup(lambda: setattr(prompts_pack, "PILOT_PATH", previous))
            with self.assertRaises(SystemExit) as raised:
                prompts_pack.require_pilot(stage="tree", n=100, dry_run=False, skip_pilot=False)
            self.assertIn("10 examples", str(raised.exception))
            prompts_pack.require_pilot(stage="tree", n=100, dry_run=True, skip_pilot=False)
            prompts_pack.require_pilot(stage="tree", n=10, dry_run=False, skip_pilot=False)
            prompts_pack.write_pilot_stage("tree", n=10, model="gpt-oss-20b")
            prompts_pack.require_pilot(stage="tree", n=100, dry_run=False, skip_pilot=False)


if __name__ == "__main__":
    unittest.main()
