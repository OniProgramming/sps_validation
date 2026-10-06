# sps_validation

Measures how much of the information in the Hebrew and Greek source text each of four
English translations retains, loses, alters or adds, sentence by sentence. The four are
WEB, BSB, OEB and SPS, over Genesis 1–50 and Ephesians 1–6. Each translation is measured
against its own base text, whatever its translation philosophy.

The method is in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Run

Requires Python ≥ 3.10 and `pip install -r requirements.txt`.

```sh
# Place the four input files (not in the repository): data/input/{WEB,BSB,OEB,SPS}.docx
# Set ANTHROPIC_API_KEY and OPENAI_API_KEY in the environment.

python3 -m sps_validation.run_all --pilot    # prepare + 10 trial requests per judge + cost plan (a few cents)
python3 -m sps_validation.run_all --judge    # sample sized to the budget (--budget 12 per account), asks before spending
python3 -m sps_validation.report --mock      # pipeline test on random judgements (not a result)
python3 -m unittest discover -s tests
```

Outputs in `build/report/`: `report.md` (the four article tables, then details and validity),
`table1.csv`–`table4.csv` (the article tables, both dimensions: sense conveyed and source preserved),
`sentences.csv` (one row per source sentence × translation) and `summary.json`.

Ablations and baselines (protocol v1.2, `docs/BASELINES.md`):

```sh
python3 -m sps_validation.ablation              # A1–A6 from the existing verdicts (free)
python3 -m sps_validation.baselines export      # inputs for the baselines
python3 -m sps_validation.baselines gemba gpt   # GEMBA-MQM (asks before spending); also: gemba claude
python3 -m sps_validation.baselines comet cometkiwi --set all   # needs unbabel-comet (Python ≤ 3.12) + HF token
python3 -m sps_validation.baselines report      # build/report/baselines/
```

xCOMET-XL needs a 16 GB GPU or 30 GB RAM: `notebooks/baselines_comet.ipynb` (Kaggle).

## Layout

| Path | Contents |
|---|---|
| `sps_validation/sources.py` | pinned source datasets (MACULA Hebrew/Greek, BHSA, Robinson-Pierpont, Westcott-Hort) |
| `sps_validation/ingest.py` | .docx extraction; SPS inline apparatus parsed into typed spans |
| `sps_validation/segment.py` | source-anchored sentence units; WLC↔BHS and SBLGNT↔RP/WH differences |
| `sps_validation/features.py` | information-feature inventory (LEX, ASP, STEM, VOICE, MOOD, REF, NUM, DEF, REL, NEG, ARG) |
| `sps_validation/align.py` | verse-free monotonic sentence alignment, same algorithm for all four |
| `sps_validation/judge.py` | blinded judge requests; Claude and GPT judges via batch APIs |
| `sps_validation/validate.py` | planted-error (known-answer) tests and test–retest sample |
| `sps_validation/report.py` | per-sentence scores, totals, bootstrap CIs, Friedman/Wilcoxon, Krippendorff's α |
| `sps_validation/ablation.py` | ablations A1–A6 recomputed from the existing verdicts |
| `sps_validation/baselines.py` | COMETKiwi, xCOMET, GEMBA-MQM on the same planted errors and sentences |
| `sps_validation/gemba_mqm.py` | GEMBA-MQM prompt and scoring, reproduced from the authors' code (CC BY-SA 4.0) |
| `sps_validation/plan.py` | sizes the random Genesis sample to the budget from the pilot's measured cost |
| `config/prices.json` | model prices used for the cost estimate |
| `sps_validation/run_all.py` | runs everything in order |
| `config/bases.json` | which source text each translation is measured against |
| `docs/PROTOCOL.md` | study protocol (draft) |

## Source licences

- MACULA Hebrew and MACULA Greek (Clear Bible): CC BY 4.0.
- ETCBC BHSA (BHS text): CC BY-NC 4.0.
- WLC: public domain.
- Robinson-Pierpont and Westcott-Hort texts: public domain.
