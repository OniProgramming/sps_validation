"""Extract the four translations from their .docx files into JSONL records.

WEB, BSB and OEB are verse-numbered ("Genesis 1:1  text").
SPS is numbered by paragraph inside units (D1, D2, ...); only the unit id is
kept. Its inline apparatus is kept as typed spans, and `eval_text` (the text
that is evaluated) keeps only transliterations and [[…]] ambiguity markers:

    translit     italic run that is not an annotation (a transliterated source word)
    source_form  {…}      source form given beside an English rendering
    note         {— … — …} inline explanatory note (from Gen 12 onward)
    alternatives [[a|b]]  ambiguity marker listing the readings kept open
    loss_mark    word*    rendering flagged as losing part of the source field
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from .docx_reader import Run, paragraph_text, read_paragraphs

VERSE_RE = re.compile(r"^(Genesis|Ephesians)\s+(\d+):(\d+)\s+(.*)$")
UNIT_RE = re.compile(r"^(BERESHIT|Ephesians)\s*·\s*D(\d+)$")
# The evaluated SPS text keeps transliterations and [[a|b]] ambiguity markers;
# everything in braces ({source form}, {note}, {gloss}) and the * marks are removed.
REMOVED_FROM_EVAL = {"source_form", "note", "loss_mark"}

PARA_NO_RE = re.compile(r"^\d+$")
SEPARATOR_RE = re.compile(r"^(·\s*)+$")

SPAN_PATTERNS = [
    ("source_form", re.compile(r"\{[^{}]*\}")),
    ("alternatives", re.compile(r"\[\[[^\[\]]*\]\]")),
    ("loss_mark", re.compile(r"\*")),
]


def parse_verse_file(path: str, code: str) -> list[dict]:
    records = []
    for runs in read_paragraphs(path):
        m = VERSE_RE.match(paragraph_text(runs))
        if m:
            book, ch, vs, text = m.groups()
            records.append(
                {
                    "translation": code,
                    "book": book,
                    "chapter": int(ch),
                    "verse": int(vs),
                    "text": text.strip(),
                }
            )
    return records


def english_vocabulary(translation_dir: Path) -> set[str]:
    """Word forms used by WEB, BSB and OEB — used only to tell an English gloss
    inside SPS braces from a transliterated source form."""
    vocab: set[str] = set()
    for code in ("WEB", "BSB", "OEB"):
        with (translation_dir / f"{code}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", json.loads(line)["text"]))
    return vocab


def hebrew_forms(source_dir: Path) -> set[str]:
    """ASCII-folded transliterations of every Genesis word (MACULA Hebrew)."""
    import glob
    import xml.etree.ElementTree as ET

    forms: set[str] = set()
    for path in glob.glob(str(source_dir / "macula-hebrew/WLC/lowfat/01-Gen-*-lowfat.xml")):
        for w in ET.parse(path).getroot().iter("w"):
            if w.get("transliteration"):
                forms.add(fold(w.get("transliteration")))
    return forms


def fold(word: str) -> str:
    word = unicodedata.normalize("NFD", word.lower())
    return re.sub(r"[^a-z]", "", word)


def parse_sps(
    path: str, english: set[str] | None = None, hebrew: set[str] | None = None
) -> tuple[list[dict], list[str]]:
    """SPS paragraphs in reading order, tagged only with their unit (D1, D2, …).

    Unit headers (titles, subtitles, verse ranges) are ignored. An exact repeat
    of an earlier paragraph is skipped and reported.
    """
    records: list[dict] = []
    warnings: list[str] = []
    seen: dict[str, str] = {}
    unit = None
    current: dict | None = None
    in_text = False

    def close():
        nonlocal current
        if current is not None:
            text, spans = _spans(current.pop("_runs"))
            key = f"{current['book']}:{text}"
            where = f"{current['book']} para {current['para']}"
            if key in seen:
                warnings.append(f"{where} repeats {seen[key]} word for word; skipped")
            else:
                seen[key] = where
                current["text"], current["spans"] = text, spans
                current["eval_text"], current["eval_spans"] = strip_spans(text, spans, REMOVED_FROM_EVAL)
                records.append(current)
            current = None

    for runs in read_paragraphs(path):
        line = paragraph_text(runs)
        m = UNIT_RE.match(line)
        if m:
            close()
            in_text = True
            book = "Genesis" if m.group(1) == "BERESHIT" else "Ephesians"
            unit = {"book": book, "unit": f"{book[:3].upper()}-D{int(m.group(2))}"}
            continue
        if not in_text:
            continue
        if line.startswith("Here ends") or SEPARATOR_RE.match(line):
            close()
            if line.startswith("Here ends"):
                unit = None
            continue
        if unit is None:
            continue
        if PARA_NO_RE.match(line):
            close()
            current = {"translation": "SPS", **unit, "para": int(line), "_runs": []}
            continue
        if current is not None:
            if current["_runs"]:
                current["_runs"].append(Run(" ", False, False))
            current["_runs"].extend(runs)
        # lines between a unit header and its first paragraph are titles: ignored
    close()
    if english:
        _mark_glosses(records, english, hebrew or set())
    return records, warnings


def _mark_glosses(records: list[dict], english: set[str], hebrew: set[str]) -> None:
    """`{— compassion / mercy}` is an English gloss (a note), `{— himmol}` a source form.

    A dash-introduced brace without *…* source text is a gloss when its words
    are English word forms (as used by WEB/BSB/OEB): a single word must also not
    be a Hebrew word of Genesis ({— met} is Hebrew מֵת); a phrase needs at least
    two thirds English words or SPS names. Hyphenated words are checked part by part.
    """
    translit = {
        w.lower()
        for r in records
        for s in r["spans"]
        if s["type"] == "translit"
        for w in re.findall(r"[\wʿʾ’-]+", s["text"])
    }
    for r in records:
        changed = False
        for s in r["spans"]:
            if s["type"] != "source_form" or "*" in s["text"]:
                continue
            if s["text"][1:].lstrip()[:1] not in "—–":
                continue
            words = [w.lower() for w in re.findall(r"[^\s/—–{}…\-]+", s["text"])]
            eng = [w in english and w not in translit for w in words]
            known = [e or w in translit for e, w in zip(eng, words)]
            if len(words) == 1:
                gloss = eng[0] and fold(words[0]) not in hebrew
            else:
                gloss = any(eng) and sum(known) >= 2 * len(words) / 3
            if gloss:
                s["type"] = "note"
                s["note"] = s.pop("form")
                changed = True
        if changed:
            r["eval_text"], r["eval_spans"] = strip_spans(r["text"], r["spans"], REMOVED_FROM_EVAL)


def strip_spans(text: str, spans: list[dict], kinds: set[str]) -> tuple[str, list[dict]]:
    """Remove spans of the given kinds from the text, re-basing the remaining spans."""
    cuts = sorted((s["start"], s["end"]) for s in spans if s["type"] in kinds)
    out, kept, pos, shift = [], [], 0, []
    for a, b in cuts:
        out.append(text[pos:a])
        shift.append((b, b - a))
        pos = b
    out.append(text[pos:])

    def moved(i: int) -> int:
        return i - sum(n for end, n in shift if end <= i)

    for s in spans:
        if s["type"] not in kinds:
            kept.append(s | {"start": moved(s["start"]), "end": moved(s["end"])})
    joined = "".join(out)
    # collapse the whitespace the cut leaves behind, keeping offsets consistent
    result, mapping = [], []
    for i, ch in enumerate(joined):
        if ch == " " and (not result or result[-1] == " " or (i + 1 < len(joined) and joined[i + 1] in ",.;:!?")):
            mapping.append(len(result))
            continue
        mapping.append(len(result))
        result.append(ch)
    mapping.append(len(result))
    kept = [s | {"start": mapping[s["start"]], "end": mapping[s["end"]]} for s in kept]
    return "".join(result).strip(), kept


def _classify_braces(span: dict) -> None:
    """{be} is a bare source form; {— *nakar* — the root: he knows them} is an inline note.

    Inside notes, *…* is italic markup, not a loss mark.
    """
    content = span["text"][1:-1].strip()
    bare = content.lstrip("—– ").strip()
    first = re.match(r"\*([^*]+)\*", bare)
    plain = bare.replace("*", "")
    if " — " in plain or " – " in plain or len(plain.split()) > 4:
        span["type"] = "note"
        span["form"] = first.group(1) if first else ""
        span["note"] = plain
    else:
        span["form"] = plain


def _spans(runs: list[Run]) -> tuple[str, list[dict]]:
    """Flatten runs to text and return typed apparatus spans with char offsets."""
    text = ""
    italic_mask: list[bool] = []
    for run in runs:
        text += run.text
        italic_mask += [run.italic] * len(run.text)

    # Normalise whitespace while keeping the mask aligned.
    chars, mask = [], []
    for ch, it in zip(text, italic_mask):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        mask.append(it)
    while chars and chars[0] == " ":
        chars.pop(0), mask.pop(0)
    while chars and chars[-1] == " ":
        chars.pop(), mask.pop()
    text = "".join(chars)

    spans: list[dict] = []
    taken = [False] * len(text)
    for kind, pattern in SPAN_PATTERNS:
        for m in pattern.finditer(text):
            if any(taken[m.start() : m.end()]):
                continue
            span = {"type": kind, "start": m.start(), "end": m.end(), "text": m.group(0)}
            if kind == "alternatives":
                span["options"] = [o.strip() for o in re.split(r"[|/]", m.group(0)[2:-2]) if o.strip()]
            if kind == "source_form":
                _classify_braces(span)
            if kind == "loss_mark":
                w = re.search(r"([\wʿʾ’'\-]+)$", text[: m.start()])
                span["word"] = w.group(1) if w else ""
            spans.append(span)
            for i in range(m.start(), m.end()):
                taken[i] = True

    # Remaining italic stretches are transliterations (one span per word group).
    for w in re.finditer(r"[^\s—–]+", text):
        core = re.match(r"^[^\wʿʾ]*(.*?)[^\wʿʾ]*$", w.group(0))
        cs, ce = w.start() + core.start(1), w.start() + core.end(1)
        if ce > cs and all(mask[cs:ce]) and not any(taken[cs:ce]):
            spans.append({"type": "translit", "start": cs, "end": ce, "text": text[cs:ce]})
            for i in range(cs, ce):
                taken[i] = True
    spans.sort(key=lambda s: s["start"])
    return text, spans



def main(argv: list[str]) -> None:
    input_dir = Path(argv[1] if len(argv) > 1 else "data/input")
    out_dir = Path(argv[2] if len(argv) > 2 else "build/translations")
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in ("WEB", "BSB", "OEB"):
        recs = parse_verse_file(str(input_dir / f"{code}.docx"), code)
        _write(out_dir / f"{code}.jsonl", recs)
        print(f"{code}: {len(recs)} verses")
    sps, warnings = parse_sps(
        str(input_dir / "SPS.docx"), english_vocabulary(out_dir), hebrew_forms(Path("data/sources"))
    )
    for warning in warnings:
        print("WARNING:", warning)
    _write(out_dir / "SPS.jsonl", sps)
    kinds: dict[str, int] = {}
    for r in sps:
        for s in r["spans"]:
            kinds[s["type"]] = kinds.get(s["type"], 0) + 1
    print(f"SPS: {len(sps)} paragraphs, apparatus spans {kinds} (eval_text keeps transliterations and [[…]] only)")


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
