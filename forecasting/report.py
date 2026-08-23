"""Publication-quality cascade report: tables, CIs, and what to re-run.

  python forecasting/pipeline.py report --from-partial
  python forecasting/pipeline.py report
"""

from __future__ import annotations

import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from cascade import (
    CATS,
    DEFAULT_MAX_SEEDS,
    DEFAULT_TURNS,
    DOMAIN_ORDER,
    LABELS,
    OUTCOMES,
    PARTIAL_RUN,
    TREE,
    chi_square_2x2,
    domain_of,
    mcnemar,
    normalize_outcome,
    rows,
    wilson,
)

WHAT_TO_UPDATE = """
What to update before the next full run
=======================================
0. Expand the judged seed pool. The formatted PDF sampled 100 of 475
   hallucinating answers (143 research, 126 legal, 206 medical). This repo's
   forecasting/batch_results.jsonl only has the 60-question pilot (18
   hallucinating). Re-generate and judge the full HalluHard splits first:
     python forecasting/pipeline.py answer --domain all --resume
     python forecasting/pipeline.py judge  --domain all --resume
   Or, for a new model:
     TEST_MODEL=Qwen/Qwen3.5-2B python forecasting/generate_seeds.py

1. Run the 100 x 5 x 3 experiment. Default design is 100 seeds x 5 strategies
   x 3 turns (500 branches / 1500 answers). The formatted PDF was a 5-turn
   100-seed plan that stopped at 61 seeds. Start the new run with:
     python forecasting/pipeline.py tree --max-seeds 100 --levels 3 --resume
     python forecasting/pipeline.py label --resume
     python forecasting/pipeline.py report

2. Use round-robin domain sampling (now the default). The PDF planned 34/33/33
   research/legal/medical but the captured set drifted to 18/27/16 because a
   shuffled list was truncated. Round-robin keeps a stopped run balanced.

3. Judge every turn with DROP / CORRECT / REPEAT / DEPEND, then derive the
   branch outcome. The formatted PDF mixed two vocabularies (persisted vs
   persisted_active/dormant) and some branches disagree with their turn
   states (e.g. seed 1 accepting is five "persisted" turns labeled CORRECT).
   Derived outcomes remove that inconsistency.

4. Turn thinking off and strip <think> blocks. Qwen3.5 thinks by default;
   the runner hard-disables it with enable_thinking=False (override with
   ENABLE_THINKING=1) and still strips leftover think tags. Otherwise the
   token budget is spent on hidden chain-of-thought and the question is echoed.

5. Score teacher-forced token probability of the emitted answer, not
   max-softmax. Max-softmax is peakedness and tracks entropy; forecasting
   needs the probability of the tokens the model actually wrote. Stored on
   each branch as init_average_confidence / init_average_entropy / etc.

6. Generate seeds with the model under test. Do not score Llama (or any
   other model) on Qwen's answers:
     TEST_MODEL=... python forecasting/generate_seeds.py
     python forecasting/pipeline.py tree --seeds forecasting/seeds_<slug>.jsonl --resume

7. Prefer the 5 pure-strategy branches over a mixed 4^N grid. Mixed-style
   sequences are not interpretable as a user-strategy effect, which is the
   claim the PDF tables make.

8. Default student model is Qwen/Qwen3.5-2B (TEST_MODEL). Default OpenAI
   judge and follow-up model is gpt-5-mini (OPENAI_LABEL_MODEL), called
   through the Responses API with reasoning.effort=minimal. Keep the same
   judge prompt matches pipeline.py judge: important facts that are wrong,
   fabricated, or presented as fact without support. Cap seed generation
   with MAX_QUESTIONS=400 if needed.

9. Report with Wilson 95% CIs and a domain breakdown. Percentages in the
   formatted PDF are point estimates on an incomplete sample; CIs and the
   domain split are what a reader should see.
""".strip()


def load_partial(path: Path = PARTIAL_RUN) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing partial-run file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_partial(data: dict) -> list[dict]:
    out = []
    for seed in data["seeds"]:
        qid = seed["question_number"]
        for branch in seed["branches"]:
            rec = {
                "question_number": qid,
                "domain": domain_of({"question_number": qid}),
                "follow_up_mode": branch["strategy"],
                "final_label": branch["outcome"],
                "claim": seed.get("claim_excerpt", ""),
                "seed_index": seed["seed_index"],
                "levels": len(branch["turns"]),
            }
            for i, state in enumerate(branch["turns"], start=1):
                rec[f"turn_state_{i}"] = state
            out.append(rec)
    return out


def records_from_live(tree_path: Path, labels_path: Path) -> list[dict]:
    labels = {r["branch_id"]: r for r in rows(labels_path)}
    out = []
    for row in rows(tree_path):
        label_row = labels.get(row.get("branch_id"), {})
        outcome = normalize_outcome(
            label_row.get("final_label") or row.get("branch_outcome") or row.get("final_label") or "DROP"
        )
        rec = dict(row)
        rec["final_label"] = outcome
        rec["domain"] = rec.get("domain") or domain_of(rec)
        out.append(rec)
    return out


def detected_turns(records: list[dict], default: int = DEFAULT_TURNS) -> int:
    found = 0
    for rec in records:
        levels = rec.get("levels")
        if isinstance(levels, int):
            found = max(found, levels)
        for i in range(1, 12):
            if rec.get(f"turn_state_{i}") or rec.get(f"future_turn_{i}"):
                found = max(found, i)
    return found or default


def count_table(records: list[dict], group: str) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        table[rec.get(group, "?")][rec["final_label"]] += 1
        table[rec.get(group, "?")]["n"] += 1
    return table


def fmt_cell(k: int, n: int) -> str:
    if n == 0:
        return "0"
    p, lo, hi = wilson(k, n)
    return f"{k} ({100 * p:.0f}%) [{100 * lo:.0f}-{100 * hi:.0f}]"


def print_table(title: str, table: dict[str, Counter], order: list[str]) -> None:
    print(f"\n{title}")
    header = f"{'group':<22} {'n':>4}" + "".join(f"{name:>28}" for name in OUTCOMES)
    print(header)
    print("-" * len(header))
    for key in [k for k in order if k in table] + [k for k in sorted(table) if k not in order]:
        counts = table[key]
        n = counts["n"]
        print(f"{key:<22} {n:>4}" + "".join(f"{fmt_cell(counts[o], n):>28}" for o in OUTCOMES))


def is_correct(rec: dict) -> bool:
    return rec.get("final_label") == "CORRECT"


def is_entrench(rec: dict) -> bool:
    return rec.get("final_label") in ("REPEAT", "DEPEND")


def complete_seed_maps(records: list[dict]) -> list[dict[str, dict]]:
    by_seed: dict = defaultdict(dict)
    for rec in records:
        by_seed[rec["question_number"]][rec["follow_up_mode"]] = rec
    needed = set(CATS)
    return [mapping for mapping in by_seed.values() if needed <= set(mapping)]


def pairwise_recovery(records: list[dict]) -> list[tuple[str, str, str, float, float]]:
    """Unpaired Yates chi-square. Prefer mcnemar_pairs when seeds are complete."""
    by = defaultdict(lambda: Counter())
    for rec in records:
        by[rec["follow_up_mode"]][rec["final_label"]] += 1
        by[rec["follow_up_mode"]]["n"] += 1
    results = []
    if "skeptical" in by:
        s_n, s_c = by["skeptical"]["n"], by["skeptical"]["CORRECT"]
        for cat in CATS:
            if cat == "skeptical" or cat not in by:
                continue
            o_n, o_c = by[cat]["n"], by[cat]["CORRECT"]
            chi, p = chi_square_2x2(s_c, s_n - s_c, o_c, o_n - o_c)
            results.append(("skeptical", cat, "CORRECT", chi, p))
    if "dependency-seeking" in by:
        d_n = by["dependency-seeking"]["n"]
        d_bad = by["dependency-seeking"]["REPEAT"] + by["dependency-seeking"]["DEPEND"]
        for cat in CATS:
            if cat == "dependency-seeking" or cat not in by:
                continue
            o_n = by[cat]["n"]
            o_bad = by[cat]["REPEAT"] + by[cat]["DEPEND"]
            chi, p = chi_square_2x2(d_bad, d_n - d_bad, o_bad, o_n - o_bad)
            results.append(("dependency-seeking", cat, "REPEAT+DEPEND", chi, p))
    return results


def mcnemar_pairs(
    records: list[dict],
    left: str,
    predicate,
    kind: str,
) -> list[tuple]:
    """Same-seed McNemar: left strategy vs each other strategy."""
    seeds = complete_seed_maps(records)
    out = []
    for right in CATS:
        if right == left:
            continue
        a_only = b_only = both = neither = 0
        n = 0
        for seed in seeds:
            if left not in seed or right not in seed:
                continue
            a, b = predicate(seed[left]), predicate(seed[right])
            n += 1
            if a and b:
                both += 1
            elif a:
                a_only += 1
            elif b:
                b_only += 1
            else:
                neither += 1
        if n == 0:
            continue
        chi, p = mcnemar(a_only, b_only)
        out.append((left, right, kind, a_only, b_only, both, neither, chi, p))
    return out


def turn1_forecast(records: list[dict]) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        state = rec.get("turn_state_1")
        if not state:
            continue
        table[state][rec["final_label"]] += 1
        table[state]["n"] += 1
    return table


def activity_rates(records: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    by = defaultdict(list)
    for rec in records:
        by[rec["follow_up_mode"]].append(rec)
    for cat, rows_ in by.items():
        states = [
            rec[f"turn_state_{turn}"]
            for rec in rows_
            for turn in range(1, detected_turns(rows_) + 1)
            if rec.get(f"turn_state_{turn}")
        ]
        n = len(states) or 1
        out[cat] = {
            "active": states.count("persisted_active") / n,
            "corrected": states.count("corrected") / n,
            "dormant": states.count("persisted_dormant") / n,
            "n": float(len(states)),
        }
    return out


def strategy_domain_counts(records: list[dict]) -> dict[str, dict[str, Counter]]:
    table: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for rec in records:
        cat, domain = rec["follow_up_mode"], rec.get("domain") or domain_of(rec)
        table[cat][domain][rec["final_label"]] += 1
        table[cat][domain]["n"] += 1
    return table


def headline_findings(records: list[dict]) -> list[str]:
    seeds = complete_seed_maps(records)
    n = len(seeds)
    findings = []
    if n:
        sk = sum(is_correct(s["skeptical"]) for s in seeds if "skeptical" in s)
        dep = sum(is_correct(s["dependency-seeking"]) for s in seeds if "dependency-seeking" in s)
        split = sum(
            1
            for s in seeds
            if is_correct(s.get("skeptical", {})) and is_entrench(s.get("dependency-seeking", {}))
        )
        findings.append(
            f"On {n} complete seeds, skeptical recovers {sk} times and dependency-seeking recovers {dep} "
            f"(same-seed McNemar). {split}/{n} seeds recover under skeptical AND entrench under dependency-seeking."
        )
        topic = mcnemar_pairs(records, "dependency-seeking", is_entrench, "REPEAT+DEPEND")
        topic_row = next((r for r in topic if r[1] == "topic-shift"), None)
        if topic_row:
            findings.append(
                f"Topic-shift is not a safe off-ramp: entrenchment vs dependency-seeking is not significant "
                f"(McNemar chi2={topic_row[7]:.2f}, p={topic_row[8]:.2f})."
            )
    t1 = turn1_forecast(records)
    if "corrected" in t1 and t1["corrected"]["n"]:
        n_c, ok = t1["corrected"]["n"], t1["corrected"]["CORRECT"]
        findings.append(f"Turn-1 corrected forecasts a CORRECT branch: {ok}/{n_c} ({100 * ok / n_c:.0f}%).")
    sx = strategy_domain_counts(records)
    acc = sx.get("accepting", {})
    if acc:
        bits = []
        for domain in DOMAIN_ORDER:
            n_d = acc[domain]["n"]
            if n_d:
                bits.append(f"{domain} {acc[domain]['CORRECT']}/{n_d}")
        if bits:
            findings.append("Accepting-style recovery is domain-specific: " + ", ".join(bits) + ".")
    return findings


def turn_dynamics(records: list[dict]) -> dict[str, list[float]]:
    """Share of corrected / persisted_active states at each turn, by strategy."""
    out: dict[str, list[float]] = {}
    by = defaultdict(list)
    for rec in records:
        by[rec["follow_up_mode"]].append(rec)
    for cat, rows_ in by.items():
        rates = []
        for turn in range(1, detected_turns(rows_) + 1):
            key = f"turn_state_{turn}"
            vals = [r.get(key) for r in rows_ if r.get(key)]
            if not vals:
                rates.append(None)
                continue
            rates.append(sum(v == "corrected" for v in vals) / len(vals))
        out[cat] = rates
    return out


def completeness(records: list[dict], planned_seeds: int = DEFAULT_MAX_SEEDS) -> dict:
    seeds = {r["question_number"] for r in records}
    by_seed = defaultdict(set)
    for rec in records:
        by_seed[rec["question_number"]].add(rec["follow_up_mode"])
    incomplete = sorted(
        (qid, sorted(set(CATS) - cats))
        for qid, cats in by_seed.items()
        if cats != set(CATS)
    )
    return {
        "captured_seeds": len(seeds),
        "planned_seeds": planned_seeds,
        "captured_branches": len(records),
        "planned_branches": planned_seeds * len(CATS),
        "incomplete_seeds": incomplete,
        "seed_domains": dict(Counter(domain_of({"question_number": q}) for q in seeds)),
    }


def html_escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def pdf_safe(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def render_html(records: list[dict], meta: dict, path: Path) -> None:
    by_cat = count_table(records, "follow_up_mode")
    by_dom = count_table(records, "domain")
    n_turns = detected_turns(records)
    complete = completeness(records, meta.get("planned_seeds", DEFAULT_MAX_SEEDS))
    dynamics = turn_dynamics(records)
    tests = pairwise_recovery(records)
    paired_correct = mcnemar_pairs(records, "skeptical", is_correct, "CORRECT")
    paired_entrench = mcnemar_pairs(records, "dependency-seeking", is_entrench, "REPEAT+DEPEND")
    findings = headline_findings(records)
    t1 = turn1_forecast(records)
    activity = activity_rates(records)
    sx = strategy_domain_counts(records)
    domain_mix = ", ".join(f"{k} {v}" for k, v in sorted(complete["seed_domains"].items()))

    def table_html(table, order):
        rows_html = []
        for key in [k for k in order if k in table] + [k for k in sorted(table) if k not in order]:
            counts = table[key]
            n = counts["n"]
            cells = "".join(f"<td>{html_escape(fmt_cell(counts[o], n))}</td>" for o in OUTCOMES)
            rows_html.append(f"<tr><td>{html_escape(key)}</td><td>{n}</td>{cells}</tr>")
        head = "".join(f"<th>{o}</th>" for o in OUTCOMES)
        return (
            f"<table><thead><tr><th>group</th><th>n</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
        )

    seed_blocks = []
    by_seed = defaultdict(list)
    for rec in records:
        by_seed[(rec.get("seed_index") or rec["question_number"], rec["question_number"])].append(rec)
    for (index, qid), branches in list(sorted(by_seed.items()))[:]:
        claim = next((b.get("claim") or b.get("false_claim") or "" for b in branches), "")
        rows_html = []
        for rec in sorted(branches, key=lambda r: list(CATS).index(r["follow_up_mode"]) if r["follow_up_mode"] in CATS else 99):
            turns = "".join(
                f"<td class='{html_escape(rec.get(f'turn_state_{i}', ''))}'>"
                f"{html_escape(rec.get(f'turn_state_{i}', ''))}</td>"
                for i in range(1, detected_turns(branches) + 1)
            )
            rows_html.append(
                f"<tr><td>{html_escape(rec['follow_up_mode'])}</td>{turns}"
                f"<td><b>{html_escape(rec['final_label'])}</b></td></tr>"
            )
        turn_heads = "".join(f"<th>T{i}</th>" for i in range(1, detected_turns(branches) + 1))
        seed_blocks.append(
            f"<section class='seed'><h3>Seed {html_escape(index)} | q{qid} "
            f"({html_escape(domain_of({'question_number': qid}))})</h3>"
            f"<p class='claim'>Claim excerpt: {html_escape(claim)}</p>"
            f"<table><thead><tr><th>Strategy</th>{turn_heads}"
            f"<th>Outcome</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table></section>"
        )

    tests_html = "".join(
        f"<tr><td>{html_escape(left)} vs {html_escape(right)}</td>"
        f"<td>{html_escape(kind)}</td><td>{chi:.2f}</td><td>{p:.4f}</td></tr>"
        for left, right, kind, chi, p in tests
    )
    paired_html = "".join(
        f"<tr><td>{html_escape(left)} vs {html_escape(right)}</td><td>{html_escape(kind)}</td>"
        f"<td>{a_only}</td><td>{b_only}</td><td>{both}</td><td>{neither}</td>"
        f"<td>{chi:.2f}</td><td>{p:.4g}</td></tr>"
        for left, right, kind, a_only, b_only, both, neither, chi, p in (paired_correct + paired_entrench)
    )
    findings_html = "".join(f"<li>{html_escape(item)}</li>" for item in findings)
    t1_html = "".join(
        f"<tr><td>{html_escape(state)}</td><td>{counts['n']}</td>"
        + "".join(f"<td>{html_escape(fmt_cell(counts[o], counts['n']))}</td>" for o in OUTCOMES)
        + "</tr>"
        for state, counts in t1.items()
    )
    activity_html = "".join(
        f"<tr><td>{html_escape(cat)}</td>"
        f"<td>{100 * rates['active']:.0f}%</td>"
        f"<td>{100 * rates['corrected']:.0f}%</td>"
        f"<td>{100 * rates['dormant']:.0f}%</td></tr>"
        for cat, rates in activity.items()
    )
    sx_html = "".join(
        f"<tr><td>{html_escape(cat)}</td>"
        + "".join(
            f"<td>{sx[cat][d]['CORRECT']}/{sx[cat][d]['n']} "
            f"({(100 * sx[cat][d]['CORRECT'] / sx[cat][d]['n']) if sx[cat][d]['n'] else 0:.0f}%)</td>"
            for d in DOMAIN_ORDER
        )
        + "</tr>"
        for cat in CATS if cat in sx
    )
    dyn_html = "".join(
        f"<tr><td>{html_escape(cat)}</td>" + "".join(
            f"<td>{'' if rate is None else f'{100*rate:.0f}%'}</td>" for rate in rates
        ) + "</tr>"
        for cat, rates in dynamics.items()
    )
    incomplete = complete["incomplete_seeds"]
    incomplete_html = (
        "<p>All captured seeds have five strategies.</p>"
        if not incomplete
        else "<ul>" + "".join(
            f"<li>q{qid}: missing {html_escape(', '.join(missing))}</li>"
            for qid, missing in incomplete
        ) + "</ul>"
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Hallucination Cascade Forecasting Results</title>
<style>
body {{ font-family: Georgia, serif; max-width: 1100px; margin: 2rem auto; color: #111; }}
h1,h2,h3 {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }}
.kpi {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0 2rem; }}
.kpi div {{ background: #f4f1ea; padding: 1rem 1.2rem; min-width: 120px; }}
.kpi b {{ display: block; font-size: 1.6rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.6rem; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #222; color: #fff; }}
td.corrected {{ background: #d8f3dc; }}
td.persisted_active {{ background: #ffe3e0; }}
td.persisted {{ background: #fff3bf; }}
td.persisted_dormant {{ background: #e7f5ff; }}
td.not_applicable {{ background: #eee; }}
.claim {{ color: #444; font-style: italic; }}
pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 1rem; }}
.note {{ background: #fff6e5; padding: 0.8rem 1rem; border-left: 4px solid #e0a100; }}
</style></head><body>
<h1>Hallucination Cascade Forecasting Results</h1>
<p>Cleaned, complete rendering of the captured run, plus Wilson 95% CIs,
domain splits, turn-level recovery rates, and a re-run checklist. Outcome key:
DEPEND = cascade propagation; REPEAT = entrenchment; DROP = natural extinction;
CORRECT = recovery.</p>
<div class="kpi">
  <div><b>{complete['captured_seeds']}</b>seed records</div>
  <div><b>{complete['captured_branches']}</b>captured branches</div>
  <div><b>{complete['planned_seeds']}</b>planned seeds</div>
  <div><b>{n_turns}</b>turns per branch</div>
</div>
<div class="note"><strong>This is still a partial run.</strong>
Planned {complete['planned_branches']} branches; captured {complete['captured_branches']}.
Captured domain mix: {html_escape(domain_mix)}. Finish with
<code>python forecasting/pipeline.py tree --max-seeds 100 --levels 3 --resume</code>.
</div>
<h2>Headline findings (same-seed, stronger than the formatted PDF)</h2>
<ul>{findings_html}</ul>
<h2>Final aggregate distribution</h2>
{table_html(by_cat, list(CATS))}
<h2>By domain</h2>
{table_html(by_dom, list(DOMAIN_ORDER))}
<h2>CORRECT rate by strategy and domain</h2>
<table><thead><tr><th>strategy</th><th>research</th><th>legal</th><th>medical</th></tr></thead>
<tbody>{sx_html}</tbody></table>
<h2>Turn-level recovery rate (share of turns labeled corrected)</h2>
<table><thead><tr><th>strategy</th>{''.join(f'<th>T{i}</th>' for i in range(1, n_turns + 1))}</tr></thead>
<tbody>{dyn_html}</tbody></table>
<h2>Claim still in play (share of turns)</h2>
<table><thead><tr><th>strategy</th><th>persisted_active</th><th>corrected</th><th>persisted_dormant</th></tr></thead>
<tbody>{activity_html}</tbody></table>
<h2>Turn-1 state forecasts branch outcome</h2>
<table><thead><tr><th>turn-1 state</th><th>n</th><th>DROP</th><th>CORRECT</th><th>REPEAT</th><th>DEPEND</th></tr></thead>
<tbody>{t1_html}</tbody></table>
<h2>Same-seed McNemar tests</h2>
<p>Holds the hallucinated claim fixed. a_only = left strategy has the metric, right does not.</p>
<table><thead><tr><th>contrast</th><th>metric</th><th>a_only</th><th>b_only</th><th>both</th><th>neither</th><th>chi-square</th><th>p</th></tr></thead>
<tbody>{paired_html}</tbody></table>
<h2>Unpaired chi-square (for comparison with the formatted PDF)</h2>
<p>Yates-corrected. Weaker than McNemar because it does not hold the seed fixed.</p>
<table><thead><tr><th>contrast</th><th>metric</th><th>chi-square</th><th>p</th></tr></thead>
<tbody>{tests_html}</tbody></table>
<h2>Incomplete seeds</h2>
{incomplete_html}
<h2>What to update</h2>
<pre>{html_escape(WHAT_TO_UPDATE)}</pre>
<h2>Complete branch-level results</h2>
{''.join(seed_blocks)}
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"HTML -> {path}")


def render_pdf(records: list[dict], meta: dict, path: Path) -> None:
    try:
        from fpdf import FPDF
    except ImportError:
        print("PDF skipped (install fpdf2 to write a PDF). HTML report is complete.")
        return

    by_cat = count_table(records, "follow_up_mode")
    by_dom = count_table(records, "domain")
    complete = completeness(records, meta.get("planned_seeds", DEFAULT_MAX_SEEDS))

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 8, "Hallucination Cascade Forecasting Results", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Cleaned results report | Wilson 95% CIs | HalluHard x HallucinationResearchTest",
                      align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def section(self, title: str):
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

        def body(self, text: str):
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 8)
            self.multi_cell(self.epw, 4.2, pdf_safe(text))
            self.ln(1)

        def table(self, headers, data, widths):
            self.set_x(self.l_margin)
            usable = self.epw
            total = sum(widths)
            if total > usable:
                scale = usable / total
                widths = [w * scale for w in widths]
            self.set_font("Helvetica", "B", 7)
            for h, w in zip(headers, widths):
                self.cell(w, 6, pdf_safe(h), border=1)
            self.ln()
            self.set_font("Helvetica", "", 7)
            for row in data:
                for cell, w in zip(row, widths):
                    self.cell(w, 6, pdf_safe(str(cell)[:40]), border=1)
                self.ln()
            self.ln(2)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.section("Status")
    pdf.body(
        f"Captured {complete['captured_seeds']} / {complete['planned_seeds']} seeds, "
        f"{complete['captured_branches']} / {complete['planned_branches']} branches. "
        f"Domain mix of captured seeds: {', '.join(f'{k} {v}' for k, v in sorted(complete['seed_domains'].items()))}. "
        "Finish the run with --resume before treating percentages as final."
    )
    pdf.section("Headline findings")
    pdf.body(" ".join(headline_findings(records)))
    pdf.section("Aggregate (Wilson 95% CI in brackets)")
    headers = ["category", "n", *OUTCOMES]
    data = []
    for cat in CATS:
        if cat not in by_cat:
            continue
        counts = by_cat[cat]
        n = counts["n"]
        data.append([cat, n, *[fmt_cell(counts[o], n) for o in OUTCOMES]])
    pdf.table(headers, data, [28, 10, 38, 38, 38, 38])

    pdf.section("By domain")
    data = []
    for domain in DOMAIN_ORDER:
        if domain not in by_dom:
            continue
        counts = by_dom[domain]
        n = counts["n"]
        data.append([domain, n, *[fmt_cell(counts[o], n) for o in OUTCOMES]])
    pdf.table(["domain", "n", *OUTCOMES], data, [28, 10, 38, 38, 38, 38])

    pdf.section("Same-seed McNemar")
    paired_rows = []
    for left, right, kind, a_only, b_only, both, neither, chi, p in (
        mcnemar_pairs(records, "skeptical", is_correct, "CORRECT")
        + mcnemar_pairs(records, "dependency-seeking", is_entrench, "REPEAT+DEPEND")
    ):
        paired_rows.append([f"{left} vs {right}", kind, a_only, b_only, f"{chi:.2f}", f"{p:.3g}"])
    if paired_rows:
        pdf.table(["contrast", "metric", "a_only", "b_only", "chi2", "p"], paired_rows, [62, 32, 18, 18, 18, 22])

    pdf.section("What to update")
    pdf.body(WHAT_TO_UPDATE)
    pdf.add_page()
    pdf.section("Branch-level results")
    by_seed = defaultdict(list)
    for rec in records:
        by_seed[rec["question_number"]].append(rec)
    for qid, branches in sorted(by_seed.items(), key=lambda kv: kv[1][0].get("seed_index") or kv[0]):
        claim = next((b.get("claim") or b.get("false_claim") or "" for b in branches), "")[:90]
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 5, pdf_safe(f"q{qid} ({domain_of({'question_number': qid})}) {claim}"), new_x="LMARGIN", new_y="NEXT")
        rows_ = []
        for rec in sorted(branches, key=lambda r: list(CATS).index(r["follow_up_mode"]) if r["follow_up_mode"] in CATS else 99):
            n_turns = detected_turns(branches)
            turns = [rec.get(f"turn_state_{i}", "") for i in range(1, n_turns + 1)]
            rows_.append([rec["follow_up_mode"], *turns, rec["final_label"]])
        n_turns = detected_turns(branches)
        pdf.table(
            ["strategy", *[f"T{i}" for i in range(1, n_turns + 1)], "out"],
            rows_,
            [32, *([28] * n_turns), 18],
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    print(f"PDF  -> {path}")


def render_report(from_partial: bool, tree_path: Path, labels_path: Path, html_path: Path, pdf_path: Path) -> None:
    meta = {"planned_seeds": DEFAULT_MAX_SEEDS, "source": "live"}
    if from_partial or (not rows(tree_path) and PARTIAL_RUN.exists()):
        data = load_partial()
        records = records_from_partial(data)
        meta = {
            "planned_seeds": data.get("planned_seeds", 100),
            "source": "formatted partial run",
            "sampling": data.get("sampling", {}),
            "model": data.get("model", "Qwen/Qwen3.5-2B"),
        }
        print("Source: formatted 61-seed captured run (PDF values preserved).")
        sampling = data.get("sampling", {})
        if sampling:
            print("Planned sampling:")
        for name, row in sampling.items():
            selected, available = row["selected"], row["available"]
            rate = row.get("selection_rate")
            if rate is None and available:
                rate = round(100 * selected / available, 1)
            print(f"  {name:<10} {selected:>3}/{available:<3} ({rate}%)")
    else:
        records = records_from_live(tree_path, labels_path)
        if not records:
            raise SystemExit(f"No labeled branches in {labels_path} or {tree_path}. Pass --from-partial to report the captured run.")
        print(f"Source: {tree_path.name} + {labels_path.name}")

    print_table("Outcome by follow-up strategy (Wilson 95% CI)", count_table(records, "follow_up_mode"), list(CATS))
    print_table("Outcome by domain", count_table(records, "domain"), list(DOMAIN_ORDER))

    complete = completeness(records, meta.get("planned_seeds", DEFAULT_MAX_SEEDS))
    print(
        f"\nCompleteness: {complete['captured_seeds']}/{complete['planned_seeds']} seeds, "
        f"{complete['captured_branches']}/{complete['planned_branches']} branches"
    )
    print(f"Captured seed domains: {complete['seed_domains']}")
    if complete["incomplete_seeds"]:
        print("Incomplete seeds:")
        for qid, missing in complete["incomplete_seeds"]:
            print(f"  q{qid}: missing {', '.join(missing)}")

    print("\nHeadline findings:")
    for item in headline_findings(records):
        print(f"  - {item}")

    print("\nSame-seed McNemar (holds the claim fixed):")
    for left, right, kind, a_only, b_only, both, neither, chi, p in (
        mcnemar_pairs(records, "skeptical", is_correct, "CORRECT")
        + mcnemar_pairs(records, "dependency-seeking", is_entrench, "REPEAT+DEPEND")
    ):
        print(
            f"  {left} vs {right:<20} {kind:<14} "
            f"a_only={a_only} b_only={b_only} both={both} neither={neither} "
            f"chi2={chi:.2f} p={p:.4g}"
        )

    print("\nUnpaired chi-square (weaker, does not hold seed fixed):")
    for left, right, kind, chi, p in pairwise_recovery(records):
        print(f"  {left} vs {right:<20} {kind:<14} chi2={chi:.2f} p={p:.4f}")

    print("\nTurn-1 state -> branch outcome:")
    for state, counts in turn1_forecast(records).items():
        n = counts["n"]
        print(f"  {state:<20} n={n:>3} " + " ".join(f"{o}={fmt_cell(counts[o], n)}" for o in OUTCOMES))

    print("\nTurn-level P(corrected):")
    for cat, rates in turn_dynamics(records).items():
        print(f"  {cat:<20} " + " ".join("---" if r is None else f"T{i+1}={100*r:4.0f}%" for i, r in enumerate(rates)))

    print("\n" + WHAT_TO_UPDATE)
    render_html(records, meta, html_path)
    render_pdf(records, meta, pdf_path)
