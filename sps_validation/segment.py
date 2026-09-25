"""Cut the source texts into sentence units — the unit of evaluation.

Units are defined on the source side only, so that every translation is
measured against the same set of units regardless of how it punctuates
its English.

Hebrew (WLC, MACULA Hebrew syntax trees)
    MACULA stores one tree per verse. Where the verse root is a
    coordination of independent clauses, each clause is one sentence;
    a clause-initial conjunction (וְ etc.) belongs to the clause it opens.
    Material that is not a clause attaches to the preceding sentence.

Greek (SBLGNT, MACULA Greek syntax trees)
    MACULA sentences, which follow the SBLGNT editors' sentence
    punctuation (period, question mark, raised dot).

For Ephesians the Robinson-Pierpont and Westcott-Hort texts are aligned
word-by-word to SBLGNT inside each verse, so every sentence unit also
carries the RP and WH wording and the list of textual differences.
"""

from __future__ import annotations

import csv
import difflib
import glob
import json
import re
import sys
import unicodedata
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

HEB_FIELDS = [
    "ref", "unicode", "lemma", "strongnumberx", "morph", "pos", "class", "type",
    "stem", "person", "gender", "number", "state", "gloss", "english",
    "sdbh", "sensenumber", "lexdomain", "coredomain", "role", "transliteration", "after",
    "frame", "participantref", "subjref",
]
GRK_FIELDS = [
    "ref", "unicode", "lemma", "strong", "morph", "class", "type", "person", "number",
    "gender", "case", "tense", "voice", "mood", "degree", "gloss", "english",
    "domain", "ln", "role", "after", "frame", "subjref", "referent",
]


# --------------------------------------------------------------------------- Hebrew

def hebrew_units(macula_dir: Path) -> list[dict]:
    units = []
    for path in sorted(glob.glob(str(macula_dir / "WLC/lowfat/01-Gen-*-lowfat.xml"))):
        for verse in ET.parse(path).getroot().findall("sentence"):
            ref = verse.get("id")
            groups = _split_clauses([c for c in verse if c.tag in ("wg", "w")])
            for i, nodes in enumerate(groups, 1):
                words = sorted((w for n in nodes for w in _words(n)), key=lambda w: w.get(XML_ID))
                units.append(_unit(f"{ref}#{i}", "hbo", [ref], words, HEB_FIELDS, _role_map(nodes)))
    return units


def _split_clauses(nodes: list[ET.Element]) -> list[list[ET.Element]]:
    flat = []
    for n in nodes:
        flat.extend(_expand(n))
    groups: list[list[ET.Element]] = []
    prefix: list[ET.Element] = []
    for n in flat:
        if n.tag == "wg" and n.get("class") == "cl":
            groups.append(prefix + [n])
            prefix = []
        elif n.tag == "w" and n.get("class") == "cj":
            prefix.append(n)
        elif groups and not prefix:
            groups[-1].append(n)
        else:
            prefix.append(n)
    if prefix:
        if groups:
            groups[-1].extend(prefix)
        else:
            groups.append(prefix)
    return groups


def _expand(node: ET.Element) -> list[ET.Element]:
    """Unwrap class-less coordination groups that contain clauses."""
    if node.tag == "wg" and node.get("class") is None:
        children = [c for c in node if c.tag in ("wg", "w")]
        if any(c.tag == "wg" and (c.get("class") == "cl" or c.get("class") is None) for c in children):
            out = []
            for c in children:
                out.extend(_expand(c))
            return out
    return [node]


def parse_bhsa(tf_dir: Path, book: str = "Genesis", abbrev: str = "GEN") -> dict[str, list[dict]]:
    """BHS words of one book from ETCBC Text-Fabric files: {'GEN 1:1': [{'text', 'ref'}]}."""
    otype = _tf(tf_dir / "otype.tf")
    max_slot = max(n for n, t in otype.items() if t == "word")
    books, chapters, verses = (_tf(tf_dir / f"{f}.tf") for f in ("book", "chapter", "verse"))
    wanted = {n for n, t in otype.items() if t == "verse" and books.get(n) == book}
    slots = _tf(tf_dir / "oslots.tf", start=max_slot, only=wanted)
    words = _tf(tf_dir / "g_word_utf8.tf")
    out: dict[str, list[dict]] = {}
    for n in sorted(wanted, key=lambda n: _slot_list(slots[n])[0]):
        ref = f"{abbrev} {chapters[n]}:{verses[n]}"
        out[ref] = [
            {"ref": ref, "text": words[s]} for s in _slot_list(slots[n]) if _norm(words.get(s, ""))
        ]
    return out


def _tf(path: Path, start: int = 0, only: set[int] | None = None) -> dict[int, str]:
    """Minimal Text-Fabric .tf reader (node-feature and oslots files)."""
    vals: dict[int, str] = {}
    node, header = start, True
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if header:
                header = line != ""
                continue
            if "\t" in line:
                key, value = line.split("\t", 1)
                if "-" in key:
                    a, b = map(int, key.split("-"))
                    if only is None:
                        vals.update(dict.fromkeys(range(a, b + 1), value))
                    node = b
                    continue
                node = int(key)
            else:
                node, value = node + 1, line
            if only is None or node in only:
                vals[node] = value
    return vals


def _slot_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


# --------------------------------------------------------------------------- Greek

def greek_units(macula_dir: Path, rp_csv: Path, wh_file: Path) -> list[dict]:
    root = ET.parse(macula_dir / "SBLGNT/lowfat/10-ephesians.xml").getroot()
    units = []
    for n, sentence in enumerate(root.iter("sentence"), 1):
        words = sorted(_words(sentence), key=lambda w: w.get(XML_ID))
        refs = list(dict.fromkeys(w.get("ref").split("!")[0] for w in words))
        units.append(_unit(f"EPH-S{n:03d}", "grc", refs, words, GRK_FIELDS, _role_map([sentence])))
    _attach_other_editions(units, {"RP": parse_rp(rp_csv), "WH": parse_wh(wh_file)})
    return units


def parse_rp(path: Path) -> dict[str, list[dict]]:
    verses = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref = f"EPH {row['chapter']}:{row['verse']}"
            verses[ref] = _parsed_tokens(row["text"].split(), ref, lambda s: s)
    return verses


ROBINSON_TRANSLIT = str.maketrans(
    "abgdezhqiklmnxoprstufcywv", "αβγδεζηθικλμνξοπρστυφχψως"
)


def parse_wh(path: Path) -> dict[str, list[dict]]:
    verses: dict[str, list[str]] = {}
    ref = None
    for tok in path.read_text(encoding="utf-8").split():
        if re.fullmatch(r"\d+:\d+", tok):
            ref = f"EPH {tok}"
            verses[ref] = []
        elif ref:
            verses[ref].append(tok)
    return {
        r: _parsed_tokens(_wh_text_reading(toks), r, lambda s: s.translate(ROBINSON_TRANSLIT))
        for r, toks in verses.items()
    }


def _wh_text_reading(toks: list[str]) -> list[str]:
    """Keep WH's text reading: in `| A | B |` A is the text, B the margin; drop `(1:11)` notes."""
    out, option = [], 0
    for t in toks:
        if t == "|":
            option = {0: 1, 1: 2, 2: 0}[option]
        elif re.fullmatch(r"\(\d+:\d+\)", t):
            continue
        elif option in (0, 1):
            out.append(t)
    return out


def _parsed_tokens(toks: list[str], ref: str, to_greek) -> list[dict]:
    """Robinson format: word strong [strong] {MORPH} ... ; [ ] marks WH doubtful words."""
    out: list[dict] = []
    bracket = False
    for t in toks:
        if t.startswith("{"):
            out[-1]["morph"] = t.strip("{}")
        elif re.fullmatch(r"\d+\]?", t):
            out[-1].setdefault("strong", t.rstrip("]"))
            if t.endswith("]"):
                bracket = False
        else:
            opens = t.startswith("[")
            closes = t.endswith("]")
            word = t.strip("[]")
            bracket = bracket or opens
            out.append({"ref": ref, "text": to_greek(word), "bracketed": bracket})
            if closes:
                bracket = False
    return out


def _attach_other_editions(units: list[dict], editions: dict[str, dict[str, list[dict]]]) -> None:
    """Align each edition to SBLGNT chapter by chapter (so versification differences
    are not mistaken for variants) and record every difference on the unit it falls in."""
    by_chapter: dict[str, list[tuple[int, dict]]] = {}
    for ui, u in enumerate(units):
        for tok in u["tokens"]:
            by_chapter.setdefault(tok["ref"].split(":")[0], []).append((ui, tok))
    for u in units:
        u["editions"] = {name: [] for name in editions}
        u["variants"] = {name: [] for name in editions}
    for name, verses in editions.items():
        other_by_chapter: dict[str, list[dict]] = {}
        for ref, toks in verses.items():
            other_by_chapter.setdefault(ref.split(":")[0], []).extend(toks)
        for chapter, sbl in by_chapter.items():
            other = other_by_chapter.get(chapter, [])
            a = [_norm(t["text"]) for _, t in sbl]
            b = [_norm(t["text"]) for t in other]
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                # Words of the other edition go to the unit of the SBLGNT word they match
                # (or, for insertions, the SBLGNT word just before them).
                anchor = i1 if i2 > i1 else max(i1 - 1, 0)
                ui, anchor_tok = sbl[min(anchor, len(sbl) - 1)]
                units[ui]["editions"][name].extend(other[j1:j2])
                if op == "equal":
                    continue
                ours, theirs = [t for _, t in sbl[i1:i2]], other[j1:j2]
                units[ui]["variants"][name].append(
                    {
                        "ref": anchor_tok["ref"].split("!")[0],
                        "op": op,
                        "kind": _variant_kind(ours, theirs),
                        "base": " ".join(t["text"] for t in ours),
                        name.lower(): " ".join(t["text"] for t in theirs),
                    }
                )
        for u in units:
            _mark_transpositions(u["variants"][name], name.lower())


def _mark_transpositions(variants: list[dict], key: str) -> None:
    """A deletion and an insertion of the same words in one unit is a change of word order."""
    for d in variants:
        if d["op"] != "delete":
            continue
        for i in variants:
            if i["op"] == "insert" and i["kind"] == "substantive" and _norm_seq(i[key]) == _norm_seq(d["base"]):
                d["kind"] = i["kind"] = "transposition"
                break


def _norm_seq(text: str) -> list[str]:
    return [_norm(w) for w in text.split()]


def _attach_consonantal_edition(units: list[dict], name: str, verses: dict[str, list[dict]]) -> None:
    """Hebrew editions differ in word segmentation (suffixes, maqaf, empty article
    slots), so they are compared letter by letter on the consonantal text, chapter
    by chapter. Every letter-level difference is recorded on the unit it falls in."""
    chapters: dict[str, list[tuple[str, int, dict]]] = {}
    for ui, u in enumerate(units):
        u.setdefault("editions", {})[name] = []
        u.setdefault("variants", {})[name] = []
        for tok in u["tokens"]:
            for ch in _norm(tok["text"]):
                chapters.setdefault(tok["ref"].split(":")[0], []).append((ch, ui, tok))
    other: dict[str, list[tuple[str, dict]]] = {}
    for ref, toks in verses.items():
        for tok in toks:
            for ch in _norm(tok["text"]):
                other.setdefault(ref.split(":")[0], []).append((ch, tok))
    for chapter, base in chapters.items():
        theirs = other.get(chapter, [])
        sm = difflib.SequenceMatcher(
            a="".join(c for c, _, _ in base), b="".join(c for c, _ in theirs), autojunk=False
        )
        placed: set[int] = set()
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            anchor = i1 if i2 > i1 else max(i1 - 1, 0)
            _, ui, tok = base[min(anchor, len(base) - 1)]
            for _, t in theirs[j1:j2]:
                if id(t) not in placed:
                    placed.add(id(t))
                    units[ui]["editions"][name].append(t)
            if op == "equal":
                continue
            words_ours = list(dict.fromkeys(id(t) for _, _, t in base[max(i1 - 1, 0) : i2 + 1]))
            ours = {id(t): t for _, _, t in base}
            units[ui]["variants"][name].append(
                {
                    "ref": tok["ref"].split("!")[0],
                    "op": op,
                    "kind": "consonantal",
                    "base_letters": sm.a[i1:i2],
                    f"{name.lower()}_letters": sm.b[j1:j2],
                    "base_context": " ".join(ours[k]["text"] for k in words_ours),
                }
            )


def _variant_kind(ours: list[dict], theirs: list[dict]) -> str:
    """'word-division' / 'orthographic' when only spelling differs (same lexemes, same parsing)."""
    if ours and theirs and "".join(_norm(t["text"]) for t in ours) == "".join(_norm(t["text"]) for t in theirs):
        return "word-division"
    if ours and len(ours) == len(theirs) and all(
        a.get("strong") == b.get("strong") and a.get("morph") == b.get("morph")
        for a, b in zip(ours, theirs)
    ):
        return "orthographic"
    return "substantive"


def _norm(word: str) -> str:
    """Compare letters only: no accents, points or cantillation; final forms folded."""
    word = unicodedata.normalize("NFD", word.lower())
    word = "".join(c for c in word if not unicodedata.combining(c))
    word = word.translate(FINAL_FORMS)
    return re.sub(r"[^\w]", "", word)


FINAL_FORMS = str.maketrans("ςךםןףץ", "σכמנפצ")


# --------------------------------------------------------------------------- shared

def _words(node: ET.Element) -> list[ET.Element]:
    return [node] if node.tag == "w" else list(node.iter("w"))


def _role_map(nodes: list[ET.Element]) -> dict[str, str]:
    """Clause-level syntactic role of each word (s, v, o, io, p, adv, pp …) from its nearest role-bearing ancestor."""
    roles: dict[str, str] = {}

    def walk(n: ET.Element, role: str | None) -> None:
        role = n.get("role") or role
        if n.tag == "w":
            if role:
                roles[n.get(XML_ID)] = role
            return
        for c in n:
            walk(c, role)

    for n in nodes:
        walk(n, None)
    return roles


def _unit(uid: str, lang: str, refs: list[str], words, fields, roles) -> dict:
    tokens = []
    for w in words:
        tok = {"id": w.get(XML_ID)}
        for f in fields:
            v = w.get(f)
            if v is not None:
                tok[f] = v
        tok["text"] = (w.text or "").strip() or tok.get("unicode", "")
        tok.setdefault("after", "")
        if "role" not in tok and roles.get(tok["id"]):
            tok["role"] = roles[tok["id"]]
        tokens.append(tok)
    text = "".join(
        t["text"] + t["after"] + (" " if lang == "grc" and not t["after"].endswith(" ") else "")
        for t in tokens
    ).strip()
    return {"unit_id": uid, "lang": lang, "refs": refs, "text": text, "tokens": tokens}


def main(argv: list[str]) -> None:
    src = Path(argv[1] if len(argv) > 1 else "data/sources")
    out = Path(argv[2] if len(argv) > 2 else "build/sources")
    out.mkdir(parents=True, exist_ok=True)
    gen = hebrew_units(src / "macula-hebrew")
    _attach_consonantal_edition(gen, "BHS", parse_bhsa(src / "bhsa/tf/2021"))
    eph = greek_units(
        src / "macula-greek",
        src / "byzantine-majority-text/csv-unicode/strongs/with-parsing/EPH.csv",
        src / "greektext-westcott-hort/parsed/EPH.UWH",
    )
    for name, units in (("GEN", gen), ("EPH", eph)):
        with (out / f"{name}.units.jsonl").open("w", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        n_tok = sum(len(u["tokens"]) for u in units)
        print(f"{name}: {len(units)} sentence units, {n_tok} source tokens")
    for code, units, base, names in (("GEN", gen, "WLC", ("BHS",)), ("EPH", eph, "SBLGNT", ("RP", "WH"))):
        for name in names:
            kinds = Counter(v["kind"] for u in units for v in u["variants"][name])
            print(f"{code}: {sum(kinds.values())} word-level differences {base}↔{name} {dict(kinds)}")


if __name__ == "__main__":
    main(sys.argv)
