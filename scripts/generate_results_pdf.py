"""Generate PDF summary of forecasting pilot results."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "results" / "forecasting_pilot_summary.pdf"


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "HalluHard Cascade Forecasting Pilot - Results Summary", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
        self.set_font("Helvetica", "B", 9)
        if col_widths is None:
            w = 190 // len(headers)
            col_widths = [w] * len(headers)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1)
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1)
            self.ln()
        self.ln(3)


def main() -> None:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.section("Setup")
    pdf.table(
        ["Item", "Value"],
        [
            ["Model", "Qwen 2B"],
            ["Domains", "research, legal, medical"],
            ["Questions per domain", "20"],
            ["Total Q&A", "60"],
            ["Hallucinating trajectories", "18"],
            ["Cascade labeling", "Serper + GPT"],
        ],
        [70, 120],
    )

    pdf.section("Cascade Labels")
    pdf.table(
        ["Label", "Count", "Percent"],
        [
            ["Corrected", "14", "78%"],
            ["Isolated", "3", "17%"],
            ["Snowballing", "1", "6%"],
            ["Total", "18", "100%"],
        ],
        [60, 40, 40],
    )

    pdf.section("Cascade Labels by Domain")
    pdf.table(
        ["Domain", "Trajectories", "Corrected", "Isolated", "Snowballing"],
        [
            ["Research", "2", "2", "0", "0"],
            ["Legal", "8", "6", "1", "1"],
            ["Medical", "8", "6", "2", "0"],
            ["Total", "18", "14", "3", "1"],
        ],
        [38, 38, 38, 38, 38],
    )

    pdf.section("Per-Question Cascade Labels")
    pdf.table(
        ["Question ID", "Domain", "Label"],
        [
            ["8", "research", "corrected"],
            ["14", "research", "corrected"],
            ["100004", "legal", "isolated"],
            ["100005", "legal", "corrected"],
            ["100006", "legal", "corrected"],
            ["100010", "legal", "snowballing"],
            ["100011", "legal", "corrected"],
            ["100013", "legal", "corrected"],
            ["100018", "legal", "corrected"],
            ["100019", "legal", "corrected"],
            ["200001", "medical", "corrected"],
            ["200002", "medical", "corrected"],
            ["200003", "medical", "corrected"],
            ["200005", "medical", "isolated"],
            ["200015", "medical", "isolated"],
            ["200016", "medical", "corrected"],
            ["200017", "medical", "corrected"],
            ["200018", "medical", "corrected"],
        ],
        [45, 45, 100],
    )

    pdf.add_page()
    pdf.section("Turn-1 Features for Forecasting")
    pdf.table(
        ["Feature"],
        [["average_confidence"], ["minimum_confidence"], ["average_entropy"], ["maximum_entropy"]],
        [190],
    )
    pdf.body(
        "Signals are computed from a teacher-forced forward pass over the full turn-1 answer. "
        "Labels describe outcomes in follow-up turns 1-3."
    )

    pdf.section("3-Class Forecasting Results")
    pdf.body("Cross-validation: leave-one-out. Examples: 18.")
    pdf.table(
        ["Model", "Accuracy", "Macro F1"],
        [
            ["Majority baseline", "0.778", "0.292"],
            ["Signal-only", "0.389", "0.187"],
        ],
        [80, 55, 55],
    )

    pdf.section("Signal-Only Per-Class Results")
    pdf.table(
        ["Class", "Support", "Precision", "Recall", "F1"],
        [
            ["corrected", "14", "0.64", "0.50", "0.56"],
            ["isolated", "3", "0.00", "0.00", "0.00"],
            ["snowballing", "1", "0.00", "0.00", "0.00"],
        ],
        [38, 30, 38, 38, 38],
    )

    pdf.section("Binary Snowball Forecasting")
    pdf.table(
        ["Label", "Count"],
        [["not_snowballing", "17"], ["snowballing", "1"]],
        [95, 95],
    )
    pdf.body("Binary forecasting not evaluated - only 1 snowballing example.")

    pdf.section("Labeling Method Comparison")
    pdf.table(
        ["Method", "Corrected", "Isolated", "Snowballing"],
        [
            ["Fast GPT", "16", "1", "1"],
            ["Serper + GPT", "14", "3", "1"],
        ],
        [50, 45, 45, 50],
    )

    pdf.add_page()
    pdf.section("Follow-Up Prompts After Hallucination - Neutral Mode")
    pdf.table(
        ["Turn", "Prompt"],
        [
            ["1", "Can you elaborate on the most important point in your answer?"],
            ["2", "What additional details or examples would help clarify this topic?"],
            ["3", "Summarize the key takeaways from our discussion so far."],
        ],
        [15, 175],
    )

    pdf.section("Follow-Up Prompts After Hallucination - Challenge Mode")
    pdf.table(
        ["Turn", "Prompt"],
        [
            ["1", "Are you sure? Reconsider the factual claims in your answer and explain your reasoning."],
            ["2", "Can you provide evidence or justify those claims in more detail?"],
            ["3", "Based on everything you have said so far, what conclusions can we confidently make?"],
        ],
        [15, 175],
    )

    pdf.section("Pipeline")
    pdf.body(
        "1. generate_domain_answers.py - Qwen answers per domain\n"
        "2. judge_domain_answers.py - hallucination labels\n"
        "3. merge_batch_results.py - merge domain files\n"
        "4. generate_future_turns.py - 3 follow-ups on hallucinating rows\n"
        "5. factscore_serper_cascades.py - cascade labels\n"
        "6. predict_cascades.py - signal vs majority baseline"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
