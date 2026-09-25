# Source-fidelity evaluation of four English translations — protocol (draft v0.1)

Status: **draft for the author's review**. Once the open decisions (§10) are settled, this
document is frozen, time-stamped (e.g. OSF pre-registration) and only then is the full
evaluation run. Any later change is logged in §11 with its reason.

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
| WEB | Protestant ed., 26 Aug 2026 | WLC | Robinson-Pierpont (Byzantine) |
| BSB | 3rd printing | WLC | SBLGNT (operational proxy for its eclectic critical text) |
| OEB | Release 2025.6, US | WLC | Westcott-Hort |
| SPS | author's manuscript | WLC | SBLGNT |

Scope: Genesis 1–50 (1,533 verses) and Ephesians 1–6 (155 verses).

Source data (all openly licensed, pinned to commit hashes in `sps_validation/sources.py`):

- **MACULA Hebrew** (WLC text, OSHB morphology, Groves/Clear syntax trees, SDBH word senses and
  semantic domains, semantic-role frames). CC BY 4.0.
- **MACULA Greek — SBLGNT** (text, morphology, syntax trees, Louw–Nida domains). CC BY 4.0.
- **Robinson-Pierpont** Byzantine Textform with parsing. Public domain.
- **Westcott-Hort** with Robinson's parsing. Public domain.

Textual-base rule: every translation is measured against its own base (table above).
Differences between editions are computed mechanically (`segment.py`: 105 SBLGNT↔RP and
12 SBLGNT↔WH word-level differences in Ephesians, classified as substantive,
transposition, orthographic or word-division). Where a translation departs from its own base
but agrees with a reading of another edition, the item is logged as a *textual-base
difference* and excluded from the fidelity score.

## 3. Unit of analysis: the source sentence

Units are defined **on the source side only**, so all four translations are scored on
exactly the same units regardless of how each one punctuates English.

- **Hebrew.** MACULA stores one syntax tree per verse. Where the verse root is a coordination
  of independent clauses, each clause is a sentence; a clause-initial conjunction belongs to
  the clause it opens; non-clausal material attaches to the preceding sentence.
  Result: **4,220 sentences, 32,365 source tokens** (tokens include prefixed particles).
  Note: quoted direct speech forms its own sentence(s), separate from the quotation formula
  ("And God said" | "Let there be light" | "and there was light").
- **Greek.** MACULA sentences, which follow the SBLGNT editors' sentence punctuation (period,
  question mark, raised dot). Result: **78 sentences, 2,416 source tokens.** Long periods
  such as Eph 1:3–6 remain one unit.

The English span for each unit is located by sentence alignment (§6). For SPS, which is
paragraph-numbered, the search is limited to the paragraphs of the unit (D1…D12, D1…D5)
covering the verse.

## 4. What counts as information: the feature inventory

Each source token carries a set of **information features**. The inventory is generated
mechanically from the source annotation *before* any translation is seen, so the judge
cannot decide what counts.

| Class | Feature | Hebrew source | Greek source |
|---|---|---|---|
| LEX | lexical sense / semantic field | SDBH sense + lexical domain | Louw–Nida domain |
| VERB | aspect/tense | qatal, yiqtol, wayyiqtol, participle, … | tense-form |
| VERB | voice / stem meaning | stem (niphal, piel, hiphil …) | voice |
| VERB | mood / modality | imperative, jussive, cohortative | mood |
| REF | person, number (verbs, pronouns); gender where it identifies a referent | morph | morph |
| NOM | number; definiteness | morph, article | morph, article |
| REL | relation expressed by the word or form | construct state, prepositions, conjunctions, particles, negation | case function, prepositions, conjunctions, particles, negation |
| ARG | who-does-what-to-whom | semantic-role frame (A0/A1…) | syntactic role |

Features that carry no information in context are excluded by fixed rules, not by the judge.
Example: the grammatical gender of an inanimate noun.

## 5. Scoring each feature

For every feature of every source token, the judge assigns one outcome from the English text:

| Outcome | Meaning | Score |
|---|---|---|
| retained | the information is recoverable from the English | 1 |
| partial | recoverable but narrowed, broadened or weakened | 0.5 |
| lost | not recoverable | 0 |
| distorted | the English conveys different information (wrong sense, time, agent, relation …) | 0, also counted as an error |
| transliterated | the source word is reproduced, not translated | see §7 |

Every outcome records the English words that render the token (the word alignment) and a
one-line justification, so each judgement can be checked.

**Additions.** English material not grounded in any source token is classified as
(a) required by English grammar (articles, auxiliaries, copula), which is neutral;
(b) explicitation of information implicit in the source, such as a pronoun's referent,
which is neutral but counted; or (c) unsupported addition, which is counted as an error.

## 6. Measures

For sentence *s* and translation *T*:

- **Retention** R = Σ scores / number of scorable features (the share of source
  information recovered). **Loss** = 1 − R.
- **Accuracy** P = supported information / (supported + distorted + unsupported additions).
- **Fidelity index** F = 2PR / (P + R).
- **Sub-scores by feature class** (LEX, VERB, REF, NOM, REL, ARG). These are the reported
  answers to "semantic field", "lexical" and "grammatical" fidelity.

Aggregation: per sentence, then per book and per translation (token-weighted, so a
two-word sentence does not count as much as a forty-word one). Uncertainty: 95% bootstrap
intervals, resampling sentences. Comparison: all four translations are scored on the same
sentences (a repeated-measures design), so the tests are the Friedman test, then pairwise
Wilcoxon signed-rank tests with Holm correction. Effect sizes are reported alongside
*p*-values.

Descriptive, **not part of the score**:
- lexical concordance, i.e. how consistently each source lemma is rendered;
- SPS disclosed-loss marks (`*`);
- textual-base differences.

## 7. SPS apparatus and transliteration

SPS carries an inline apparatus. `ingest.py` parses it deterministically from the .docx
formatting:

| Mark | Detected from | Count |
|---|---|---|
| transliteration | italic run | 1,430 |
| `{…}` bare source form | braces | 255 |
| `{— … — …}` inline explanatory note | braces with commentary | 32 |
| `[[a\|b]]` open ambiguity | double brackets | 2 |
| `word*` disclosed loss | asterisk outside braces | 22 (21 of them in Gen 1–4) |

The other three inputs were supplied **with notes and footnotes removed**. Treating SPS's
inline apparatus as text while the others lose their notes would not be symmetric. The
protocol therefore scores two pre-registered conditions:

1. **Text-only (primary).** Apparatus stripped from SPS (`{…}` removed, `[[a|b]]` → first
   option, `*` removed). Transliterations remain, because they are the running text.
2. **Text + apparatus.** SPS with its inline apparatus, and WEB/BSB/OEB with their published
   footnotes restored.

Transliterations are scored under three pre-registered rules, and all three are reported:

- **T0 (receiver-oriented).** LEX = lost unless the sentence context makes the meaning
  recoverable. Grammatical features visible in the transliterated form (for example Greek
  case endings) are judged normally.
- **T½.** LEX = partial.
- **T1 (source-oriented).** LEX = retained, since the lexeme's identity is preserved.

Proper names are exempt: every translation transliterates them.

If the ranking of translations changes between T0 and T1, that dependence is itself a
reported finding.

## 8. Neutrality safeguards

1. The protocol, feature inventory and prompts are frozen before any full run.
2. Translation names are removed and the order is randomised per sentence. SPS cannot be
   fully blinded, because its transliterations identify it; this is stated as a limitation.
3. Identical prompts, rules and features are used for all four translations.
4. Two independent judges from different model families score everything, and their
   agreement is reported.
5. **Human validation.** A stratified random sample (≈5% of sentences, all four translations)
   is scored independently by at least two Hebraists/Hellenists who do not know which
   translation is which. Krippendorff's α is reported for human–human and human–model
   agreement.
6. All per-sentence, per-feature judgements, the code and the pinned sources are published
   (with a Zenodo DOI) so that any reader can re-run or audit the result.
7. Conflict of interest: the author of this study is the translator of SPS. This is declared,
   and items 1–6 are the answer to it.

## 9. Pipeline

| Step | Module | Status |
|---|---|---|
| Fetch pinned sources | `sources.py` | done |
| Extract translations + SPS apparatus | `ingest.py` | done |
| Source sentence units + edition variants | `segment.py` | done |
| Feature inventory per source token | `features.py` | next |
| Sentence alignment (source unit → English span) | `align.py` | next |
| Judging (features, additions, word alignment) | `judge.py` | needs an LLM API key |
| Aggregation, statistics, report | `report.py` | — |

## 10. Open decisions (for the author)

1. Transliteration: which rule (T0 / T½ / T1) is the *primary* analysis, with the other two
   as sensitivity analyses? Or all three reported with equal standing?
2. Condition 2 (text + apparatus) requires the published footnotes of WEB/BSB/OEB. Include
   it, or report text-only only?
3. BSB Ephesians base: SBLGNT as proxy, or another edition?
4. OEB/WEB Genesis base: WLC assumed (both follow the Masoretic Text). Confirm.
5. Human raters: who, and how many?
6. Judges: which model families.

## 11. Data issues found in the supplied files

- **SPS Ephesians paragraphs 15 and 16 are identical** (Eph 4:17–24 appears twice). Nothing
  appears to be missing, but the duplicate must be removed or explained before the
  evaluation.
- **SPS unit GEN-D7** declares "Gen 11:27 – 12:20" in its header, but its 82 paragraphs run
  until GEN-D8 begins at 25:12, i.e. 11:27–25:11. Effective ranges are inferred from the
  next unit's start (`ingest.add_effective_ranges`).
- **SPS apparatus conventions change during Genesis.** Loss marks (`*`) are used almost only in
  Gen 1–4. From about Gen 12 onward, braces sometimes hold explanatory notes (e.g.
  `{— *roʾsh ha-miṭṭah* — or, upon the top of his staff (LXX); …}`) rather than bare source forms,
  and inside these notes `*…*` is italic markup. The parser separates the two kinds, but the
  author should decide whether inline notes belong to Book I (the text) at all.
- The text in the SPS preface also contains an italic example sentence. Extraction therefore
  starts at the first unit header.
