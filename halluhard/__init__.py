"""HalluHard assets reused by Branching Hallucinations.

This folder is upstream benchmark code and data, not the D/N/V experiment.
The experiment needs two things from here:

1. Seed questions (`research_questions`, `legal_cases`, `medical_guidelines`, `coding`)
2. Retrieval (`judging_pipeline`: Serper → fetch/PDF → filter)

The HalluHard *paper* pipeline (generate chats → HTML report) also lives here
if you need to reproduce https://arxiv.org/abs/2602.01031.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
