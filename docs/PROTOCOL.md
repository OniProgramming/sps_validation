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

**Two dimensions.** Every information feature is judged twice: *sense conveyed* (does it reach
an ordinary reader of the English?) and *source preserved* (can it be recovered from the text as
written, including through transliterations and literal renderings?). All totals, tables and
tests are reported for both.

**Validity without human raters.** The instrument is an operational, rubric-based measurement.
The checks below show consistency and sensitivity to known errors. They do not by themselves
prove that every judgement is correct, and the article should present them that way.
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

- v1.1 (after pilot 3: agreement sense 88%, source 67%; frozen for the full run):
  - *Sense conveyed* is the primary dimension. *Source preserved* is reported as a secondary
    dimension. Its lower inter-judge agreement in the pilot (67%) is stated as a limitation,
    and its results are interpreted with that in mind.
  - Budget US$16 per judge: 425 randomly sampled Genesis sentences + all 78 Ephesians
    sentences, in all four translations.
- v1.0 (after pilot 2: 10 requests × 2 judges; agreement sense 87.8%, source 57.7%):
  - The source-preserved rules were ambiguous: one judge required the Hebrew/Greek *form*
    to be visible and scored standard English equivalents ("God" for אֵל, "went" for
    wayyiqtol) as partial. The rules now state that a direct counterpart (a standard English
    equivalent, or a transliteration) preserves the item. Partial is reserved for items that
    survive only indirectly (merged, restructured, image replaced, context only).
    Distinctions English grammar cannot mark are not penalised when the directly
    corresponding English category is used. The change applies identically to all
    translations.
- v0.9 (third review; final before the pilot):
  - The planted-error generator keeps only edits that stay grammatical by construction:
    - a noun is changed only where nothing later in its clause agrees with it (it is
      followed by punctuation, the end, or a preposition/conjunction), and never when it
      has an irregular plural form (children, men…);
    - the tense shift applies only to a simple past in a main clause: no auxiliary, negation
      or subordinator ("when", "that", "who"…) earlier in the clause.
    Unsafe cases are excluded rather than repaired.
  - Budget per judge: US$16. The sample is sized with a 10% safety margin over the
    pilot's measured cost per request.
- v0.8 (second external review, before any paid run):
  - The planted-error tests are scored on **both** dimensions, so *source preserved* has its
    own sensitivity and false-alarm figures. Agreement between judges shows consistency for
    both dimensions, not correctness.
  - Note for the article: in *source preserved*, a transliteration preserves the identity of
    the source word (formal correspondence). That alone does not show that the contextual
    sense is preserved, which is what *sense conveyed* measures. The two dimensions must be
    read together.
  - Planted errors: no past → future shift after an auxiliary ("had left"); correct
    irregular plurals (wife/wives); no number flip on participles or adjectives.
  - אֱלֹהִים: whether its plural counts as number is decided by the SDBH sense ("gods",
    000397001001000), no longer by an English gloss. No gloss now influences the feature
    inventory.
  - Zero denominators are defined, so no NaN can occur. Accuracy with nothing asserted is 1;
    fidelity is 0 when retention is 0.
- v0.7 (two dimensions, fixed before any paid run):
  - Every feature receives two verdicts, both reported with equal weight in all tables:
    - **Sense conveyed**: is the information conveyed to an ordinary reader of the English
      text?
    - **Source preserved**: can the information be recovered from the text as written
      (English words, literal renderings, transliterated source words, markers), even if
      that takes effort or knowledge of the source?
  - Reason: the two criteria correspond to different translation aims. Measuring only one
    would measure only one aim. Both are fixed before any results exist, and neither was
    chosen on the basis of which translation it favours.
  - Planted-error tests and rule cross-checks use the sense verdict. Judge agreement and
    test–retest are reported for both verdicts.
- v0.6 (external code review, before any paid run):
  - The judges receive **no English glosses**. MACULA's Greek `gloss` comes from the Berean
    Interlinear Bible, and the BSB is one of the translations evaluated. The Hebrew glosses
    (Cherith) are removed too, for symmetry. Judges get the form, lemma, parsing, Strong's number
    and semantic-domain codes, and must determine meaning from the source language.
    Glosses remain in two places only: the aligner (to locate text) and the planted-error
    builder (to find the English word to alter). Neither place scores anything.
  - Scoring fix: a feature's outcome counts are weighted by the judges that actually answered
    it. Before, a refusal by one judge halved the counts but not the scores, which inflated
    accuracy.
  - English with no aligned source sentence is no longer dropped. It is judged together with
    the preceding group (where it can be listed as an addition) and counted in the report.
  - Planted errors are built so that the English stays grammatical: attributive adjectives
    only; number flips only inside prepositional phrases, with correct inflection; tense
    shift past → future ("said" → "will say"). The agent/patient swap is not used, because
    too few grammatical swaps exist in all four translations. Each planted error has an
    unperturbed control judged in the same run, so sensitivity and false-alarm rate are
    measured separately. The number of cases is equalised across translations.
  - A saved run is resumed only if model, settings, instructions, schema and requests are
    identical (fingerprint). Otherwise it is refused. Every result records the model reported
    by the API.
  - Wording: the instrument is an operational, rubric-based measurement. Agreement between
    judges shows consistency, not correctness. The planted-error tests show sensitivity to
    specific error types. Neither establishes validity by itself.
  - Pilot 1 (with glosses) is superseded; a new pilot is run with the final instructions.
- v0.5 (budget):
  - Judges changed to the smaller models of the same two companies: Claude Haiku 4.5 and
    GPT-5-mini (reasoning effort medium). The reason is cost.
  - Design changed from the full corpus to a random sample:
    - all of Ephesians (78 sentences), plus a simple random sample of Genesis sentences
      (fixed seed 20260925), in all four translations;
    - the sample size is the largest that fits a fixed budget per judge (`plan.py`),
      computed from the per-request cost measured on the pilot;
    - totals and tests are computed on the sampled sentences only, so every translation is
      scored on identical sentences.
  - Planted errors: 10 per error type per translation (280 in total). Test–retest: 10% of
    the main requests.
  - Judge reasons are shortened to at most 12 words.
- v0.4 (after pilot 1: 20 requests, both judges, 90% item-level agreement):
  - Judge rules clarified where the two judges disagreed systematically. The clarifications
    apply identically to every translation, and none was based on which translation scored
    higher:
    - REF: gender or number distinctions that English cannot mark are not required;
    - NUM: English plural-form nouns and collective singulars count as the same count;
      adverbial nouns carry no count;
    - STEM/VOICE: an English verb that has the stem's meaning suffices;
    - fixed expressions: scored by what the words contribute, while live images count
      as information;
    - REL: a relation conveyed by clause order or sentence structure is retained;
    - proper names: spelling and vocalisation differences don't matter.
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
