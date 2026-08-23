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

    def test_parse_falls_back_to_word_match(self):
        self.assertEqual(parse_judge_label("I think this is REPEAT not drop"), "repeat")

    def test_display_states_match_the_formatted_pdf(self):
        self.assertEqual(display_state("depend"), "persisted_active")
        self.assertEqual(display_state("repeat"), "persisted")
        self.assertEqual(display_state("drop"), "persisted_dormant")
        self.assertEqual(display_state("correct"), "corrected")
        self.assertEqual(display_state("???"), "not_applicable")

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


class DesignDefaultTests(unittest.TestCase):
    def test_default_run_is_hundred_seeds_and_three_turns(self):
        self.assertEqual(DEFAULT_MAX_SEEDS, 100)
        self.assertEqual(DEFAULT_TURNS, 3)

    def test_default_models_are_qwen35_2b_and_gpt5_mini(self):
        self.assertEqual(DEFAULT_TEST_MODEL, "Qwen/Qwen3.5-2B")
        self.assertEqual(DEFAULT_OPENAI_JUDGE, "gpt-5-mini")
        self.assertFalse(ENABLE_THINKING)
        from runtime import _uses_responses_api
        self.assertTrue(_uses_responses_api("gpt-5-mini"))
        self.assertFalse(_uses_responses_api("gpt-4o-mini"))

    def test_seed_judge_treats_unsupported_facts_as_hallucinations(self):
        from generate_seeds import SEED_JUDGE_TEMPLATE, parse_seed_judgement
        self.assertIn("without support", SEED_JUDGE_TEMPLATE)
        self.assertIn("wrong, fabricated", SEED_JUDGE_TEMPLATE)
        label, _ = parse_seed_judgement("Overall label: Hallucinating\nReason: invented citation")
        self.assertEqual(label, "Hallucinating")


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
            self.assertIn("q100018", text)
            self.assertIn("Same-seed McNemar", text)
            self.assertNotIn("skeptical vs dependency-seeking vs", text)
            self.assertIn("dependency-seeking vs topic-shift", text)
            self.assertIn("Turn-1 state forecasts", text)
            self.assertTrue(html_path.exists())
            self.assertTrue(pdf_path.exists())

    def test_dry_run_tree_writes_five_strategy_branches(self):
        import argparse
        from pipeline import cmd_tree
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tree.jsonl"
            args = argparse.Namespace(
                categories="all",
                seeds=str(Path(__file__).resolve().parent / "batch_results.jsonl"),
                max_seeds=2,
                levels=3,
                out=str(out),
                resume=False,
                dry_run=True,
                model="Qwen/Qwen3.5-2B",
            )
            cmd_tree(args)
            lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 10)
            self.assertTrue(all(row["levels"] == 3 for row in lines))
            self.assertEqual({row["follow_up_mode"] for row in lines}, set(CATS))
            self.assertTrue(all(row["branch_outcome"] == "DROP" for row in lines))
            self.assertTrue(all(row.get("domain") in {"research", "legal", "medical"} for row in lines))
            self.assertTrue(all("turn_label_1" in row for row in lines))
            self.assertTrue(all(row.get("enable_thinking") is False for row in lines))


class TopicShiftTests(unittest.TestCase):
    def test_topic_shift_is_a_first_class_strategy(self):
        self.assertIn("topic-shift", CATS)
        self.assertEqual(
            check("Setting that aside, what is the most common misconception here?", "topic-shift", ENTITIES),
            "",
        )

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


if __name__ == "__main__":
    unittest.main()
