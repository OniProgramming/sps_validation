# sps_validation

Measures how much of the information in the Hebrew and Greek source text each of four
English translations retains, loses, alters or adds, sentence by sentence. The four are
WEB, BSB, OEB and SPS, over Genesis 1–50 and Ephesians 1–6. Each translation is measured
against its own base text, whatever its translation philosophy.

The method is in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Run

Requires Python ≥ 3.10; the steps so far use only the standard library.

```sh
# 1. Place the four input files (not in the repository):
#    data/input/{WEB,BSB,OEB,SPS}.docx

python3 -m sps_validation.sources   # fetch WLC/SBLGNT/BHS/RP/WH data at pinned commits → data/sources/
python3 -m sps_validation.ingest    # translations → build/translations/*.jsonl (SPS: notes removed)
python3 -m sps_validation.segment   # source sentence units + edition differences → build/sources/
python3 -m sps_validation.features  # information features per source word → build/features/
python3 -m sps_validation.align     # source sentence ↔ English alignment, verse-free → build/align/
python3 -m unittest discover -s tests
```

## Layout

| Path | Contents |
|---|---|
| `sps_validation/sources.py` | pinned source datasets (MACULA Hebrew/Greek, BHSA, Robinson-Pierpont, Westcott-Hort) |
| `sps_validation/ingest.py` | .docx extraction; SPS inline apparatus parsed into typed spans |
| `sps_validation/segment.py` | source-anchored sentence units; WLC↔BHS and SBLGNT↔RP/WH differences |
| `sps_validation/features.py` | information-feature inventory (LEX, ASP, STEM, VOICE, MOOD, REF, NUM, DEF, REL, NEG, ARG) |
| `sps_validation/align.py` | verse-free monotonic sentence alignment, same algorithm for all four |
| `config/bases.json` | which source text each translation is measured against |
| `docs/PROTOCOL.md` | study protocol (draft) |

## Source licences

- MACULA Hebrew and MACULA Greek (Clear Bible): CC BY 4.0.
- ETCBC BHSA (BHS text): CC BY-NC 4.0.
- WLC: public domain.
- Robinson-Pierpont and Westcott-Hort texts: public domain.
