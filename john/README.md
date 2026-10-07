# SATE on John 1 (exploratory extension)

This folder runs the same SATE instrument on a third text, John 1. Nothing in `sps_validation/` is
changed or copied. Every step calls the unchanged functions of the main study:
- `segment`: the units are MACULA sentences;
- `features`: the inventory of information items;
- `align`: the verse-free aligner;
- `judge`: the instructions, schema, models, settings and Batch APIs;
- `report`: the scores, bootstrap, Friedman/W, Wilcoxon-Holm and the article tables.

Only the inputs and the output folder (`build/john/`) differ.

| | John 1 | As in the main study? |
|---|---|---|
| Greek | SBLGNT (MACULA Greek); RP and WH for the WEB and OEB bases | Same repositories and commits |
| WEB | eBible corpus, `engwebp`, pinned commit | Same text: identical on all 155 Ephesians verses |
| BSB | `data/input/john/bsb.txt` (official download) | Same printing. Without the file, the eBible copy is used (an earlier printing) and a warning is shown |
| OEB | Release 2025.6, US spelling, pinned commit | Same edition |
| SPS | `data/input/john/SPS_John1.docx` (or `.txt`) | Read with the main study's own `ingest` functions: transliterations from italics, braces and `*` removed, `[[…]]` kept. Verse numbers and ◊ removed. The aligner gets it by paragraph, without verse numbers |
| Bases | WEB → RP, OEB → WH, BSB and SPS → SBLGNT | Same as for the New Testament |

## What differs from the main study

- **SPS transliterations.** With the `.docx`, they are read from the italics, exactly as in the
  main study. Only with a plain `.txt` are they recognised automatically:
  - a word with a diacritic (archē);
  - or a word with no part found in the WEB/BSB/OEB vocabulary (kosmos).

  In both cases they are only alignment keys. The judges decide by themselves what is
  transliterated.
- **Alignment accuracy.** It is now measured for SPS too, from its inline verse numbers. This is
  a diagnostic only: the alignment itself does not use them.
- **Instrument checks.** As in the main study, three sets are judged:
  - all 57 sentences of John 1, with no sample;
  - planted errors, each with its unperturbed control, built by the same functions with the same
    settings (`validate.build_perturbations` with `plan.PER_CELL` = 10, same seed);
  - a 10% retest.

  John 1 has few candidates for some error types: 2 adjective drops and 3 number flips per
  translation. As in the main study, every translation gets the same number of cases. The
  rule cross-checks (NEG, NUM) and the agreement between the judges are reported too.
- **Status.** The text and the analysis were added after the main results. This is an
  exploratory extension, not part of the preregistered protocol.

## Running it

```
python -m john.prepare                 # sources, units, items, alignment, requests (free)
python -m john.run                     # shows the cost, asks for "yes", judges, writes the report
python -m john.run --judges gpt        # one judge only
python -m john.run --skip-prepare      # resume after an interruption
python -m john.report                  # report again from the saved results
python -m john.run --skip-prepare --mock   # pipeline test with random judgements
```

The API keys are needed in the environment: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

Results are written to `build/john/report/`:
- `report.md` and `table1–4.csv`;
- `sentences.csv`, which contains the translations' text;
- `summary.json`.

`data/input/john/` and `build/john/judge/` contain the SPS text. They are not committed.

## Ablations and baselines

The same ablations (A1–A6) and baselines (COMETKiwi, xCOMET, GEMBA-MQM) as for Genesis and
Ephesians. They run the unchanged `sps_validation.ablation` and `sps_validation.baselines` on John 1's
folders. Step by step: `john/PASI_BASELINES.md`.

```
python -m john.ablation                            # free → build/john/report/ablation/
python -m john.baselines export                    # free → build/john/baselines/items.jsonl …
python -m john.baselines gemba gpt --set pairs     # ≈ $1.8 (asks yes)
python -m john.baselines gemba gpt --set neutral   # ≈ $0.3
# COMET models: notebooks/baselines_comet.ipynb on Kaggle with John's items.jsonl → scores in build/john/baselines/scores/
python -m john.baselines report                    # → build/john/report/baselines/
```

