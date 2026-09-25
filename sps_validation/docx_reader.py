"""Minimal .docx reader that keeps run-level formatting.

Only the standard library is used so that the extraction step has no
third-party dependency that could change its behaviour between runs.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass(frozen=True)
class Run:
    text: str
    italic: bool
    colored: bool


def read_paragraphs(path: str) -> list[list[Run]]:
    """Return every non-empty paragraph as a list of formatted runs."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    paragraphs: list[list[Run]] = []
    for p in root.iter(f"{_W}p"):
        runs: list[Run] = []
        for r in p.iter(f"{_W}r"):
            text = "".join(
                (t.text or "") if t.tag == f"{_W}t" else "\t" if t.tag == f"{_W}tab" else ""
                for t in r
            )
            if not text:
                continue
            rpr = r.find(f"{_W}rPr")
            italic = _flag(rpr, "i")
            colored = rpr is not None and rpr.find(f"{_W}color") is not None
            runs.append(Run(text, italic, colored))
        if "".join(run.text for run in runs).strip():
            paragraphs.append(_merge(runs))
    return paragraphs


def paragraph_text(runs: list[Run]) -> str:
    return re.sub(r"\s+", " ", "".join(r.text for r in runs)).strip()


def _flag(rpr, name: str) -> bool:
    if rpr is None:
        return False
    el = rpr.find(f"{_W}{name}")
    if el is None:
        return False
    return el.get(f"{_W}val", "true") not in ("0", "false")


def _merge(runs: list[Run]) -> list[Run]:
    """Join adjacent runs that share formatting (Word splits runs arbitrarily)."""
    merged: list[Run] = []
    for run in runs:
        if merged and (merged[-1].italic, merged[-1].colored) == (run.italic, run.colored):
            merged[-1] = Run(merged[-1].text + run.text, run.italic, run.colored)
        else:
            merged.append(run)
    return merged
