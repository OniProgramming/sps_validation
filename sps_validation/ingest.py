"""Extract the four translations from their .docx files into JSONL records.

WEB, BSB and OEB are verse-numbered ("Genesis 1:1  text").
SPS is numbered by paragraph inside units (D1, D2, ...) whose header gives
the verse range; its inline apparatus is kept as typed spans:

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
from pathlib import Path

from .docx_reader import Run, paragraph_text, read_paragraphs

VERSE_RE = re.compile(r"^(Genesis|Ephesians)\s+(\d+):(\d+)\s+(.*)$")
UNIT_RE = re.compile(r"^(BERESHIT|Ephesians)\s*·\s*D(\d+)$")
RANGE_RE = re.compile(
    r"\b(Gen(?:esis)?|Ephesians)\s+(\d+)[.:](\d+)\s*[–-]\s*(?:(\d+)[.:])?(\d+)\b"
)
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


def parse_sps(path: str) -> list[dict]:
    paragraphs = read_paragraphs(path)
    records: list[dict] = []
    unit = None
    header_lines: list[str] = []
    current: dict | None = None
    in_text = False

    def close():
        nonlocal current
        if current is not None:
            current["text"], current["spans"] = _spans(current.pop("_runs"))
            records.append(current)
            current = None

    for runs in paragraphs:
        line = paragraph_text(runs)
        m = UNIT_RE.match(line)
        if m:
            close()
            in_text = True
            book = "Genesis" if m.group(1) == "BERESHIT" else "Ephesians"
            unit = {"book": book, "unit": f"{book[:3].upper()}-D{int(m.group(2))}", "range": None}
            header_lines = []
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
            if unit["range"] is None:
                unit["range"] = _parse_range(" ".join(header_lines), unit["book"])
                unit["header"] = header_lines[:]
            current = {
                "translation": "SPS",
                "book": unit["book"],
                "unit": unit["unit"],
                "unit_range": unit["range"],
                "unit_header": unit["header"],
                "para": int(line),
                "_runs": [],
            }
            continue
        if current is None:
            header_lines.append(line)
        else:
            if current["_runs"]:
                current["_runs"].append(Run(" ", False, False))
            current["_runs"].extend(runs)
    close()
    return records


def add_effective_ranges(sps: list[dict], verses: list[tuple[str, int, int]]) -> list[str]:
    """Units run from their declared start to the verse before the next unit's start.

    The declared end in a unit header is kept for reference; where it disagrees
    with the effective end a warning is returned.
    """
    index = {v: i for i, v in enumerate(verses)}
    units: dict[str, dict] = {}
    for r in sps:
        units.setdefault(r["unit"], r["unit_range"] | {"unit": r["unit"]})
    ordered = list(units.values())
    warnings = []
    effective = {}
    for i, u in enumerate(ordered):
        start = (u["book"], *u["start"])
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        if nxt and nxt["book"] == u["book"]:
            end = verses[index[(nxt["book"], *nxt["start"])] - 1]
        else:
            end = max((v for v in verses if v[0] == u["book"]), key=lambda v: index[v])
        effective[u["unit"]] = {"book": u["book"], "start": list(start[1:]), "end": list(end[1:])}
        if list(end[1:]) != u["end"]:
            warnings.append(
                f"{u['unit']}: header declares {u['start']}–{u['end']}, "
                f"but the next unit implies it ends at {list(end[1:])}"
            )
    for r in sps:
        r["effective_range"] = effective[r["unit"]]
    return warnings


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


def _parse_range(text: str, book: str):
    m = RANGE_RE.search(text)
    if not m:
        return None
    _, c1, v1, c2, v2 = m.groups()
    c1, v1, v2 = int(c1), int(v1), int(v2)
    c2 = int(c2) if c2 else c1
    return {"book": book, "start": [c1, v1], "end": [c2, v2]}


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
    verses: list[tuple[str, int, int]] = []
    for code in ("WEB", "BSB", "OEB"):
        recs = parse_verse_file(str(input_dir / f"{code}.docx"), code)
        _write(out_dir / f"{code}.jsonl", recs)
        print(f"{code}: {len(recs)} verses")
        verses = verses or [(r["book"], r["chapter"], r["verse"]) for r in recs]
    sps = parse_sps(str(input_dir / "SPS.docx"))
    for warning in add_effective_ranges(sps, verses):
        print("WARNING:", warning)
    _write(out_dir / "SPS.jsonl", sps)
    kinds: dict[str, int] = {}
    for r in sps:
        for s in r["spans"]:
            kinds[s["type"]] = kinds.get(s["type"], 0) + 1
    print(f"SPS: {len(sps)} paragraphs, apparatus spans {kinds}")


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
