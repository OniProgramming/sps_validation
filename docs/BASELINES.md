# Baselines and ablations (protocol addendum v1.2)

Added at a reviewer's request after the main results (protocol v1.1) had been seen. The main
results are unchanged. Everything below is computed from the same frozen sets
(`build/judge/sets/`) and, for SATE, from the same judge verdicts. The definitions in this file
were fixed before any baseline was run.

## 1. Ablations (`python -m sps_validation.ablation`, no cost)

Each variant recomputes the same quantities from the same categorical verdicts. Nothing is
re-judged.

| | Variant | What changes |
|---|---|---|
| A1 | Binary taxonomy | Preserved / not preserved, with partial counted either as preserved or as not preserved. There is no DISTORTED category, so distortions count as loss. |
| A2 | Dimension | Sense only, source only, or both (the mean of the two fidelities). |
| A3 | Feature family | Leave one out: lexical (LEX); verbal (ASP, STEM, VOICE, MOOD); referential (REF, NUM, DEF); relational (REL, ARG, NEG). |
| A4 | Judges | Claude only, GPT only, or combined (the mean of the judges, as in the primary scoring). |
| A5 | Partial weight | w = 0.25 and 0.75 (primary: 0.5). |
| A6 | Accuracy | Without unsupported additions; LOST and DISTORTED collapsed (D = 0). |

For each variant, book and dimension the script reports:
- the fidelity of each translation;
- their order;
- Kendall's W;
- the number of significant pairs (Wilcoxon, Holm);
- how many of the 12 pairwise conclusions (6 pairs × 2 books) differ from the primary analysis.

For A1 and A4 it also recomputes the planted-error detection and false-alarm rates. The
single-judge, full-taxonomy rows reproduce `summary.json` exactly; this is tested.

## 2. Baselines (`python -m sps_validation.baselines`)

| Method | Model | Mode | Output |
|---|---|---|---|
| COMETKiwi | `Unbabel/wmt22-cometkiwi-da` | Quality estimation (source + translation) | Score |
| xCOMET | `Unbabel/XCOMET-XL` | Quality estimation (no reference) | Score + error spans |
| GEMBA-MQM | Prompt and scoring of Kocmi & Federmann (2023), GEMBA commit a7a7eff | Claude Haiku 4.5 and GPT-5 mini, with the same settings as the SATE judges | MQM errors with spans, score |

**Inputs.** The baselines see exactly what SATE judged:
- the English text of each request (alignment group);
- the plain source text of its sentences: MACULA WLC/SBLGNT words, with Hebrew cantillation marks removed (vowels kept).

The COMET models read translation and source as one sequence of at most 512 tokens. Longer inputs
are cut, in the source, and the fraction of texts cut is reported.

**GEMBA-MQM models.** GEMBA-MQM was published with GPT-4. Running it here with the SATE judges'
own models isolates the protocol (feature-level, source-anchored rubric vs. segment-level MQM
annotation) from the model. Another model can be set with `--model`.

### Planted-error benchmark

There are 240 pairs: 6 error types × 4 translations × 10. Each pair is a perturbed text and its
identical, unperturbed control. They are the same pairs as in the article.

**Score-based detection** (all methods). The planted error counts as detected when
Δ = score(control) − score(perturbed) > 0. A tie is not detected.
- SATE uses the fidelity of the sentence group (sense dimension).
- Mean Δ and a one-sided Wilcoxon test of Δ > 0 are reported per error type.

**False alarms, LLM-based methods.** These verdicts vary between runs, so the false-alarm rate
is the same comparison made between two judgements of the *unchanged* text:
- SATE: control vs. the main run;
- GEMBA-MQM: control vs. a second run of the control. The script judges every control twice.

**False alarms, COMET models.** These are deterministic, so they have no such floor. For them,
the Wilcoxon test is the relevant check.

**SATE targeted detection** (as in the article). The planted error's own information item is
judged not fully retained in the perturbed text. The false alarm is the same verdict on the
control. Only cases whose item was retained in the main run count. For additions, the number of
unsupported additions has to rise above the main run. Judges are combined by the mean of their
scores, so an item counts as detected when at least one judge marks it.

**Span detection** (xCOMET, GEMBA-MQM).
- The edited words are located by a character diff between control and perturbed text, widened
  to whole words. For a deletion, the zone covers the words on both sides of the gap.
- The error is detected when a reported error span overlaps the edited words.
- For GEMBA-MQM, a quoted span is located in the text. For the two deletion types (modifier_drop,
  negation_drop), an error that quotes the deleted word also counts.
- The false alarm is a span on the same words in the control.

**Paired comparison.** An exact McNemar test compares each method with SATE (Claude+GPT,
targeted, sense) on the pairs both can assess.

### Meaning-preserving controls and matched specificity (added after the first baseline results)

The deterministic COMET models give no false-alarm rate in the planted-error benchmark, because
an identical text always gets the same score. Their detection rates were therefore not comparable
with those of methods that do have one.

This analysis was added after the first baseline results had been seen, and before any of the
texts below was scored. It uses **meaning-preserving edits** of the same controls (`neutral.jsonl`):
- one edit per text, the first that applies, in this order:
  - contraction ↔ full form ("do not" ↔ "don't", "cannot" ↔ "can't");
  - US ↔ UK spelling (toward/towards, honor/honour, …);
  - curly → straight quotation marks.
- Ambiguous forms are never touched: 's (is / has), 've, "let us".
- The same number of texts per translation is drawn with a fixed seed (26 each, 104 in all).

**Null comparisons** are comparisons between texts with the same meaning:
- the meaning-preserving edits, for every method that scored them;
- for SATE and GEMBA-MQM, whose verdicts vary between runs, two judgements of the identical text.

For each score-based method, τ is the smallest score drop that at most 5% of the null comparisons
exceed. A planted error counts as detected only when its drop exceeds τ (**detection at matched
specificity**). The report also gives:
- the share of meaning-preserving edits with *any* score drop;
- for span-based methods, how often an error span falls on the edited words of a contraction or
  spelling edit. Quote edits touch the whole quotation and are left out of this count.

```
python -m sps_validation.baselines export                                      # adds the 104 edited texts
python -m sps_validation.baselines comet cometkiwi --set neutral               # → scores/cometkiwi.neutral.jsonl
python -m sps_validation.baselines comet xcomet --set neutral --gpus 1 --half  # → scores/xcomet.neutral.jsonl
python -m sps_validation.baselines gemba gpt --set neutral                     # optional, ≈ $0.40
```

### Order of the translations

Each method also scores every main segment (1,980 groups). A segment's score is given to each of
its source sentences, so all methods are compared on the same sentences. The script reports:
- the mean per book and translation, and the resulting order;
- the Friedman W, and Wilcoxon-Holm pairs, as in Table 4;
- Kendall's τ between each baseline's four means and SATE's, for each dimension.

The scales differ, so only the orders and tests can be compared. Note that the SATE means here
are means over sentences, not the item-weighted totals of Table 1.

## 3. How to run

```
python -m sps_validation.ablation                 # A1–A6, seconds, free
python -m sps_validation.baselines export         # writes build/baselines/items.jsonl, pairs.jsonl
python -m sps_validation.baselines gemba gpt      # ≈ $2.6 for the planted errors (asks before starting)
python -m sps_validation.baselines gemba claude   # ≈ $2.3
python -m sps_validation.baselines report         # after any of the runs; uses whatever is available
```

GEMBA-MQM on the 1,980 main segments, for the ordering (`--set segments`), costs about $6–7 per
model. It is optional.

**COMET models.**
- They need `pip install unbabel-comet==2.2.7`, which requires numpy < 2, so Python 3.10–3.12.
- They need a Hugging Face account that has accepted the licence of both models (the models are
  gated), with the token in the `HF_TOKEN` environment variable, or after `huggingface-cli login`.
- COMETKiwi (2.2 GB) runs on an ordinary PC.
- xCOMET-XL (3.5B parameters, about 14 GB) needs 30 GB of RAM or a 16 GB GPU. The free Kaggle
  notebooks have both: use `notebooks/baselines_comet.ipynb`.

The notebook processes `items.jsonl`, which contains the SPS text, on the notebook provider's
servers.

```
python -m sps_validation.baselines comet cometkiwi --set all
python -m sps_validation.baselines comet xcomet --set all [--gpus 1] [--half]
```

Results are written to `build/baselines/scores/<method>.jsonl`. They are resumable, and failed
GEMBA calls are retried when the same command is run again.

## 4. Limits stated in advance

- **The baselines are segment-level quality metrics trained on modern MT data.** None was
  designed for Biblical Hebrew or Koine Greek, for long multi-verse sentences, or for
  source-preserving strategies such as transliteration. Their results show how such metrics
  behave on this material, not their quality in their intended domain.
- **The planted errors were built by SATE's pipeline around SATE's information items.** The
  targeted detection therefore has an advantage of localisation. The score-based detection puts
  every method on the same footing: did the score drop?
- **There are 10 pairs per translation and error type.** Rates per cell are indicative; the
  pooled rates and the paired tests carry the comparison.
