# FactBench prompts

Local copy of [launch/FactBench](https://huggingface.co/datasets/launch/FactBench) (CC-BY-4.0).

Pinned revision: `67d807e76651f4649a971b3eeaeb95f72b6cba40`

| CSV | Paper tier | Rows |
|---|---|---|
| `tier_1.csv` | Hard | 532 |
| `tier_2.csv` | Moderate (Medium) | 332 |
| `tier_3.csv` | Easy | 136 |

`prompts.jsonl` contains all three tiers. Branching configs load **Hard + Moderate** only, round-robin so a 10- or 100-seed run is not Hard-only. Easy is kept so it can be enabled later with `domains = ["easy"]`.

`prompt_score` / `hallucination_score` are FactBench's labels of the original LMSYS responses. They are not used as seed answers, verification labels, or DROP/RETRACT/REPEAT/DEPEND.

Refresh:

```bash
python -m branching_hallucinations.factbench
```

```
@misc{bayat2025factbenchdynamicbenchmarkinthewild,
      title={FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation},
      author={Farima Fatahi Bayat and Lechen Zhang and Sheza Munir and Lu Wang},
      year={2025},
      eprint={2410.22257},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.22257},
}
```
