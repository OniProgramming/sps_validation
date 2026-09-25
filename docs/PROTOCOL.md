# Source-fidelity evaluation of four English translations — protocol (draft v0.3)

Status: **draft**. The author's decisions have been applied (§11). The protocol is then frozen and time-stamped (e.g. OSF pre-registration),
and only after that is the full evaluation run.

The validator is **fully computational**: no human rating enters any score. Its validity is
established computationally (§8).

## 1. Research question

How much of the information carried by the Hebrew/Greek source text is recoverable from
each English translation, and how much is lost, altered, or added?

The question concerns **fidelity of information**, not translation philosophy. A dynamic
rendering that changes word class or word order but carries the same information scores
the same as a word-for-word rendering; a word-for-word rendering that picks the wrong sense
loses. Word order and part-of-speech correspondence are therefore **not** part of the score.

## 2. Material

| Translation | Edition (as supplied) | Genesis base | Ephesians base |
|---|---|---|---|
| WEB | Protestant ed., 26 Aug 2026 | BHS (Leningrad Codex) | Robinson-Pierpont (Byzantine) |
| BSB | 3rd printing | WLC | SBLGNT* |
| OEB | Release 2025.6, US | WLC (Westminster digital Leningrad Codex) | Westcott-Hort |
| SPS | author's manuscript | WLC | SBLGNT |

\* BSB's New Testament draws on several critical editions (NA, SBLGNT, ECM). This study uses
SBLGNT only, and the article must say so. NA and ECM are not openly licensed.

Scope: Genesis 1–50 and Ephesians 1–6.

Source data (openly licensed, pinned to commit hashes in `sps_validation/sources.py`):

- **MACULA Hebrew**: WLC text, OSHB morphology, Groves/Clear syntax trees, SDBH word senses
  and semantic domains, semantic-role frames. CC BY 4.0.
- **MACULA Greek (SBLGNT)**: text, morphology, syntax trees, Louw–Nida domains. CC BY 4.0.
- **ETCBC BHSA**: digital BHS, used for WEB's Genesis base. CC BY-NC 4.0.
- **Robinson-Pierpont** Byzantine Textform with parsing. Public domain.
- **Westcott-Hort** with Robinson's parsing. Public domain.

**Textual-base rule.** Every translation is measured against its own base. Differences between
editions are computed mechanically (`segment.py`):

| Comparison | Differences | Classification |
|---|---|---|
| WLC ↔ BHS, Genesis | 16 letters | all ketiv/qere-type consonant differences (ו/י, ו/ה); no word differs |
| SBLGNT ↔ RP, Ephesians | 105 words | 78 substantive, 16 transpositions, 8 orthographic, 3 word-division |
| SBLGNT ↔ WH, Ephesians | 12 words | 11 substantive, 1 orthographic |

The rich annotation (MACULA) exists only for WLC and SBLGNT, so all features are generated
from those. For a translation with another base, every unit containing an edition difference
carries that edition's wording. A feature affected by the difference is scored against the
translation's own base; where the base lacks the word, the feature is marked
*textual-base* and excluded.

## 3. Unit of analysis: the source sentence

Units are defined **on the source side only**, so all four translations are scored on exactly
the same units. **Verse numbers play no part in the evaluation.**

- **Hebrew.** MACULA's syntax trees are split into independent clauses. A clause-initial
  conjunction belongs to the clause it opens, and non-clausal material attaches to the
  preceding sentence. Result: **4,220 sentences, 32,365 source tokens.** Quoted direct speech
  forms its own sentence(s), separate from the quotation formula.
- **Greek.** MACULA sentences, following the SBLGNT editors' punctuation (period, question
  mark, raised dot). Result: **78 sentences, 2,416 source tokens.**

## 4. Sentence alignment (`align.py`)

Each translation is read as one continuous text per book. For SPS the unit markers (D1…) and
paragraph numbers are ignored, and brace content and `*` marks are removed (§7). The same algorithm is used
for all four:

1. The English is cut into pieces at . ; : ? ! and at a comma before an opening quote or a
   coordinating conjunction.
2. Each source sentence gets a bag of keys: the stemmed English glosses of its words (MACULA)
   plus a consonant skeleton of each word's transliteration. The skeleton lets SPS
   transliterations such as *Yosef* or *reshit* match the source. Each English piece gets its
   stemmed content words and the skeletons of names and transliterations.
3. Dynamic programming finds the order-preserving segmentation that maximises IDF-weighted key
   overlap, with a length prior. It allows one sentence to take 1–8 pieces, 2–4 sentences to
   share a piece, 2:2 groupings, and 1:0 / 0:1 (omission / addition).

**Aligner accuracy.** WEB, BSB and OEB carry verse numbers in their files. These are used
*after* alignment, only to check it:

| | source sentence placed in its verse | English piece placed in its verse |
|---|---|---|
| WEB Genesis | 97.3% | 97.0% |
| BSB Genesis | 95.7% | 96.1% |
| OEB Genesis | 95.7% | 96.6% |
| WEB / BSB / OEB Ephesians | 100% | 98.2% / 98.7% / 93.5% |

These figures are lower bounds, because translations legitimately move material across verse
boundaries. SPS has no verse numbers, so for SPS the report gives the distribution of alignment
scores instead:

| median alignment score | WEB | BSB | OEB | SPS |
|---|---|---|---|---|
| Genesis | 0.61 | 0.55 | 0.54 | 0.48 |
| Ephesians | 0.53 | 0.53 | 0.34 | 0.48 |

The alignment score measures word overlap with the MACULA English glosses, so it is higher
for wording that stays close to those glosses. It is used **only** to locate text and is not a
fidelity measure.

**Robustness to alignment errors.**
- When several source sentences share one English piece (n:1), they are judged together
  against that piece. Scores stay per source sentence, because every feature belongs to one
  source word.
- The judge also sees the English immediately before and after. Information found there is
  scored as retained (marked *displaced*). A boundary error between neighbours therefore
  produces neither a false "lost" nor a false "addition".

## 5. What counts as information: the feature inventory (`features.py`)

Each source token carries a set of **information features**. The inventory is generated
mechanically from the source annotation before any translation is seen, and each feature
counts once.

| Class | Meaning | Hebrew | Greek |
|---|---|---|---|
| LEX | lexical sense / semantic field | lemma + SDBH domain | lemma + Louw–Nida domain |
| ASP | aspect / tense-form | qatal, yiqtol, wayyiqtol, … | tense |
| STEM | derived-stem meaning | niphal, piel, hiphil, … (qal unmarked) | — |
| VOICE | middle / passive | — | voice ≠ active |
| MOOD | non-indicative mood | — | imperative, subjunctive, participle, infinitive … |
| REF | who is meant: person/gender/number | finite verbs, participles, pronouns, suffixes | finite verbs, pronouns |
| NUM | number of a common noun | ✓ | ✓ |
| DEF | definiteness | article | article |
| REL | relation | prepositions, conjunctions, relative, construct state, directional he, particles | prepositions, conjunctions, particles, genitive/dative case |
| NEG | negation | ✓ | ✓ |
| ARG | who does what to whom | semantic-role frames (A0, A1, …) | semantic-role frames |

These carry no feature of their own:
- grammatical agreement;
- the object marker אֵת, which is represented by ARG;
- paragogic nun;
- the number of plurals without count meaning (שָׁמַיִם, מַיִם, פָּנִים, חַיִּים, …, and
  אֱלֹהִים when it denotes God; GKC §124).

| | features | LEX | REL | REF | NUM | ASP | ARG | DEF | STEM | VOICE | MOOD | NEG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Genesis | 56,250 | 15,870 | 13,280 | 8,321 | 6,223 | 5,059 | 4,222 | 1,807 | 1,179 | — | — | 289 |
| Ephesians | 4,644 | 1,213 | 1,067 | 404 | 551 | 327 | 302 | 431 | — | 110 | 206 | 33 |

## 6. Scoring each feature and the measures

For every feature, the judge assigns one outcome from the aligned English:

| Outcome | Meaning | Score |
|---|---|---|
| retained | the information is recoverable from the English | 1 |
| partial | recoverable but narrowed, broadened or weakened | 0.5 |
| lost | not recoverable | 0 |
| distorted | the English conveys different information | 0, also counted as an error |
| not_in_base | the translation's own source edition lacks or changes the word | excluded |

Every outcome records the English words that render the token (the word alignment) and a
one-line justification, so it can be audited.

**Additions.** English material not grounded in any source token is classified as
(a) required by English grammar, which is neutral; (b) explicitation of implicit source
information, which is neutral but counted; or (c) unsupported addition, which is an error.

**Measures** for sentence *s* and translation *T*:
- **Retention** R = Σ scores / number of scorable features. **Loss** = 1 − R.
- **Accuracy** P = supported / (supported + distorted + unsupported additions).
- **Fidelity** F = 2PR / (P + R).
- **Sub-scores** by feature class.

Output: one row per sentence per translation (`sentences.csv`), then totals per book and per
translation. Totals are feature-weighted: every feature of the book counts once. Additions are
judged per alignment group and shared among its sentences in proportion to their feature counts. The statistics are:
- 95% bootstrap intervals, resampling sentences;
- the Friedman test across the four translations, then pairwise Wilcoxon signed-rank tests with
  Holm correction;
- effect sizes.

Descriptive, **not part of the score**: lexical concordance, SPS `*` marks, and textual-base
differences.

## 7. SPS text and transliteration

**Evaluated SPS text:** the running text with its transliterations and its `[[a|b]]`
ambiguity markers (2). Removed: everything in braces, i.e. 245 bare source forms, 32
commentary notes and 9 English glosses, and the 22 `*` marks.

**Transliteration is judged, not pre-decided.** The protocol fixes one criterion for every
feature of every word in every translation: *is this source information conveyed by the English
text, as written, to a reader of that text?* A transliterated word (*elohim*, *ruach*,
*charis*) is judged by this same criterion. The judge decides case by case from the text
itself (context, established English usage), exactly as it does for an English rendering.
No outcome is fixed in advance for transliterations. They are flagged (`transliterated=true`)
so that the report can show separately how transliterated words were judged. Proper names
are retained whenever the referent is identifiable, whatever the spelling.

## 8. Neutrality and computational validation (no human raters)

**Neutrality**
1. The protocol, feature inventory and prompts are frozen before the full run.
2. Translation names are removed and the order is randomised per sentence. SPS cannot be
   fully blinded, because its transliterations identify it; this is stated as a limitation.
3. Identical inputs, rules and features are used for every translation.

**Validity without human raters**
4. **Two independent judges** from different model families score everything. Their agreement
   is reported per feature class (Krippendorff's α). Disagreements are scored as the mean,
   and the main results are also reported for each judge separately.
5. **Test–retest.** A random 10% of requests per translation (1,364) is judged a second time. Stability is reported
   as α.
6. **Known-answer (perturbation) tests.** Controlled errors are injected automatically into
   the English of all four translations: word deletion, wrong-sense substitution, number flip,
   negation removal, tense shift (past → present), agent/patient swap, and unsupported addition.
   There are 40 cases per error type per translation, 1,120 in total (`validate.py`). An error
   counts as detected when the damaged feature is judged worse than in the unperturbed
   original. Only cases whose original was judged "retained" are counted. For additions, a
   new "unsupported" addition must appear. Reported: the rate at which the validator detects each error type
   (sensitivity), and its false alarms on unperturbed text (specificity). This gives a
   measured error rate for the instrument, and shows it is equally sensitive for every
   translation.
7. **Rule-based cross-checks.** Where a feature can be tested mechanically on the aligned
   English, a deterministic check is run and its agreement with the judges reported (NEG:
   negator present; NUM: noun plurality; DEF: article/determiner).
8. **Openness.** All per-sentence, per-feature judgements, the code and the pinned sources are
   published with a DOI.
9. **Conflict of interest.** The study's author is the translator of SPS. This is declared,
   and items 1–8 are the answer to it.

## 9. Pipeline

| Step | Module | Status |
|---|---|---|
| Fetch pinned sources | `sources.py` | done |
| Extract translations; SPS apparatus; notes removed | `ingest.py` | done |
| Source sentences; edition differences (BHS, RP, WH) | `segment.py` | done |
| Sentence alignment (verse-free) | `align.py` | done |
| Feature inventory | `features.py` | done |
| Judging (features, additions, word alignment) | `judge.py` | needs LLM API keys |
| Perturbation tests, cross-checks | `validate.py` | next |
| Aggregation, statistics, per-sentence + total report | `report.py` | — |

## 10. Judges

- **Judge 1: Claude Opus 5** (Anthropic, `claude-opus-5`), with structured JSON output and the
  Message Batches API.
- **Judge 2: OpenAI GPT** (default `gpt-5`, set with `JUDGE_GPT_MODEL`; reasoning effort
  high), with strict JSON-schema output and the Batch API. It gets identical instructions,
  inputs and output schema. The exact model versions used are recorded with every judgement.

Server-side refusal fallbacks are not used, so every judgement comes from the named model.
Refusals are counted and reported.

One request is one alignment group per translation: 13,636 requests per judge (Genesis
3,041–3,727 per translation; Ephesians 77–78). The instructions are in `judge.py`
(`INSTRUCTIONS`), frozen with this protocol.

## 11. Decision log

- v0.3:
  - SPS evaluated text keeps transliterations and `[[…]]` only; all brace content and `*`
    are removed.
  - Transliteration is judged by the single fidelity criterion; no pre-set T0/T½/T1 rules.
  - Judge 1 is Claude Opus 5; judge 2 is to come from a second model family.
- v0.2:
  - SPS evaluated without notes.
  - SPS unit headers ignored; only unit ids kept.
  - Verses not used in the evaluation (only to measure aligner accuracy).
  - BSB Ephesians measured against SBLGNT, to be stated in the article.
  - WEB Genesis measured against BHS; its difference from WLC is 16 ketiv/qere letters.
  - OEB and BSB Genesis confirmed as WLC.
  - No human raters; validation is computational (§8).
- Data: SPS Ephesians paragraph 16 duplicated paragraph 15 (Eph 4:17–24). The pipeline skips
  exact repeats with a warning; the author will remove it from the manuscript.
