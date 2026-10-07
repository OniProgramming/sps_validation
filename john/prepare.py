"""Prepare John 1: sources, translations, sentence units, features, alignment, judge requests
(main set, planted errors with their controls, retest).

Every step calls the unchanged SATE functions; only the inputs and the output folder
(build/john/) differ from the Genesis/Ephesians run.

Inputs
  Greek      SBLGNT (MACULA Greek), Robinson-Pierpont and Westcott-Hort — the same
             repositories and commits as the main study, John files added.
  WEB        eBible corpus (BibleNLP/ebible, pinned commit), Protestant edition (engwebp).
             Identical to the WEB file of the main study on all 155 verses of Ephesians.
  OEB        Official release 2025.6, US spelling (openenglishbible/Open-English-Bible,
             pinned commit, artifacts/us-release/usfm/43-John.usfm) — the edition of the main study.
  BSB        data/input/john/bsb.txt — the official download (https://bereanbible.com/bsb.txt),
             the printing of the main study. Without it the eBible copy is used, with a warning:
             that copy is an earlier printing (3 of 155 Ephesians verses differ).
  SPS        data/input/john/SPS_John1.txt — the author's text, verse numbers inline, ◊ between
             paragraphs. Evaluated text: verse numbers and ◊ removed; [[…]] kept, as in the main study.
             As in the main study, SPS reaches the aligner by paragraph, without verse numbers.

Bases (as in the main study for the New Testament): WEB → Robinson-Pierpont, OEB → Westcott-Hort,
BSB and SPS → SBLGNT.

    python -m john.prepare
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from sps_validation import align as A
from sps_validation import features as F
from sps_validation import judge as J
from sps_validation import segment as S
from sps_validation import sources as SRC
from sps_validation import validate as V
from sps_validation.plan import PER_CELL

SOURCES_DIR = Path("data/sources_john")
INPUT_DIR = Path("data/input/john")
BUILD = Path("build/john")
CODE, BOOK, CHAPTER = "JHN", "John", 1
TRANSLATIONS = ("WEB", "BSB", "OEB", "SPS")
BASE_EDITION = {"WEB": "RP", "OEB": "WH"}  # BSB and SPS: SBLGNT (the annotated text itself)

JOHN_SOURCES = {
    # the same repositories and commits as sps_validation.sources; only the John files are added
    "macula-greek": {"url": "https://github.com/Clear-Bible/macula-greek.git",
                     "commit": "8423afe47b9e8f24b7772e808af45c7159a6fe7e",
                     "paths": ["/SBLGNT/lowfat/04-john.xml", "/LICENSE.md"]},
    "byzantine-majority-text": {"url": "https://github.com/byztxt/byzantine-majority-text.git",
                                "commit": "27a45ff1b7be6c17ccbfeac414f3f55732ae8e28",
                                "paths": ["/csv-unicode/strongs/with-parsing/JOH.csv", "/LICENSE.txt"]},
    "greektext-westcott-hort": {"url": "https://github.com/byztxt/greektext-westcott-hort.git",
                                "commit": "91473892d7f36f1227e5c24f8d994d1da40311ad",
                                "paths": ["/parsed/JOH.UWH", "/README.md"]},
    # English translations
    "ebible": {"url": "https://github.com/BibleNLP/ebible.git",
               "commit": "c531ff2da02843ded6d09afbe29a197ab844981f",
               "paths": ["/metadata/vref.txt", "/corpus/eng-engwebp.txt", "/corpus/eng-engbsb.txt",
                         "/corpus/eng-engoebus.txt"]},
    "open-english-bible": {"url": "https://github.com/openenglishbible/Open-English-Bible.git",
                           "commit": "1965127de5c3c103af3fdbc9288c1abec5f39994",
                           "paths": ["/artifacts/us-release/usfm/43-John.usfm", "/VERSION", "/LICENSE"]},
}


def fetch_sources() -> None:
    """Pinned sparse checkouts, by the main study's own fetch routine."""
    saved = SRC.SOURCES
    try:
        SRC.SOURCES = JOHN_SOURCES
        SRC.fetch(SOURCES_DIR)
    finally:
        SRC.SOURCES = saved


# --------------------------------------------------------------------------- translations

def ebible_chapter(name: str) -> dict[int, str]:
    vref = (SOURCES_DIR / "ebible/metadata/vref.txt").read_text(encoding="utf-8").split("\n")
    lines = (SOURCES_DIR / f"ebible/corpus/{name}").read_text(encoding="utf-8").split("\n")
    out = {}
    for ref, text in zip(vref, lines):
        m = re.fullmatch(rf"{CODE} {CHAPTER}:(\d+)", ref.strip())
        if m and text.strip():
            out[int(m.group(1))] = text.strip()
    return out


SKIP_TAGS = {"id", "ide", "h", "rem", "mt", "mt1", "mt2", "mt3", "s", "s1", "s2", "ms", "r", "d", "sp", "cl",
             "toc1", "toc2", "toc3"}


def usfm_chapter(path: Path) -> dict[int, str]:
    """Verse texts of one chapter of a USFM file: headings, footnotes and cross references
    removed, character markers dropped (their text kept). Checked against the main study's
    OEB file: identical on all 155 verses of Ephesians."""
    t = path.read_text(encoding="utf-8")
    t = re.sub(r"\\f .*?\\f\*", "", t, flags=re.S)
    t = re.sub(r"\\x .*?\\x\*", "", t, flags=re.S)
    verses: dict[tuple[int, int], str] = {}
    ch = v = None
    for line in t.split("\n"):
        m = re.match(r"\\(\w+)\*?\s?(.*)$", line.strip())
        if not m:
            if v is not None and line.strip():
                verses[(ch, v)] += " " + line.strip()
            continue
        tag, rest = m.groups()
        if tag in SKIP_TAGS:
            continue
        if tag == "c":
            ch, v = int(rest.split()[0]), None
            continue
        if tag == "v":
            n, _, rest = rest.partition(" ")
            v = int(n)
            verses[(ch, v)] = ""
        if v is not None:
            verses[(ch, v)] = (verses[(ch, v)] + " " + re.sub(r"\\\+?\w+\*?", "", rest)).strip()
    return {vs: re.sub(r"\s+", " ", s).strip() for (c, vs), s in verses.items() if c == CHAPTER}


def bsb_chapter() -> tuple[dict[int, str], str]:
    path = INPUT_DIR / "bsb.txt"
    if path.exists():
        out = {}
        for line in path.read_text(encoding="utf-8-sig").split("\n"):
            m = re.match(rf"^{BOOK} {CHAPTER}:(\d+)\s+(.*)$", line.strip())
            if m:
                out[int(m.group(1))] = m.group(2).strip()
        if out:
            return out, f"official download ({path}, sha256 {_sha(path)[:16]})"
    print("WARNING: data/input/john/bsb.txt not found — using the eBible copy of the BSB, an earlier printing "
          "than the main study's (3 of 155 Ephesians verses differ).")
    return ebible_chapter("eng-engbsb.txt"), "eBible corpus copy (earlier printing)"


VERSE_NO = re.compile(r"(?:(?<=\s)|^)(\d{1,2})(?=[^\d\s])")


def sps_paragraphs() -> list[list[tuple[int, str]]]:
    """The author's text: verse numbers inline (verse 1 unnumbered), ◊ between paragraphs.
    Evaluated text = the running text without verse numbers and ◊; [[…]] is kept.
    Returns the paragraphs, each a list of (verse, text)."""
    raw = (INPUT_DIR / "SPS_John1.txt").read_text(encoding="utf-8")
    paras = []
    for block in raw.split("◊"):
        text = re.sub(r"\s+", " ", block).strip()
        if not text:
            continue
        marks = [(m.start(), m.end(), int(m.group(1))) for m in VERSE_NO.finditer(text)]
        if not marks or marks[0][0] > 0:  # text before the first number: verse 1 (unnumbered)
            marks = [(0, 0, 1)] + marks
        verses = []
        for i, (s, e, n) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            verses.append((n, text[e:end].strip()))
        paras.append(verses)
    numbers = [n for p in paras for n, _ in p]
    if numbers != list(range(1, len(numbers) + 1)):
        raise SystemExit(f"SPS_John1.txt: verse numbers not consecutive: {numbers}")
    return paras


def sps_chapter() -> dict[int, str]:
    return {n: t for p in sps_paragraphs() for n, t in p}


def english_vocabulary() -> set[str]:
    """Word forms of the three published translations (whole Bible, eBible corpus) — used,
    as in the main study, only to tell a transliterated source word from English."""
    vocab: set[str] = set()
    for name in ("eng-engwebp.txt", "eng-engbsb.txt", "eng-engoebus.txt"):
        text = (SOURCES_DIR / f"ebible/corpus/{name}").read_text(encoding="utf-8")
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", text))
    return vocab


def transliterations(text: str, vocab: set[str]) -> list[str]:
    """SPS words that are not English: a letter with a diacritic (archē, zōē) or no part
    found in the English vocabulary (kosmos, sarx). In the main study the same words were
    marked by italics in the SPS manuscript; here the text has no formatting."""
    out = []
    for w in re.findall(r"[^\W\d_][\w’'-]*", text):
        parts = [p for p in re.split(r"[-’']", w) if p]
        if any(not c.isascii() for c in w) or all(p.lower() not in vocab for p in parts):
            out.append(w)
    return out


def translations() -> dict:
    meta = {}
    web = ebible_chapter("eng-engwebp.txt")
    meta["WEB"] = f"eBible corpus, engwebp (commit {JOHN_SOURCES['ebible']['commit'][:7]})"
    oeb = usfm_chapter(SOURCES_DIR / "open-english-bible/artifacts/us-release/usfm/43-John.usfm")
    meta["OEB"] = (f"OEB release {(SOURCES_DIR / 'open-english-bible/VERSION').read_text().strip()}, "
                   f"US spelling (commit {JOHN_SOURCES['open-english-bible']['commit'][:7]})")
    bsb, meta["BSB"] = bsb_chapter()
    sps = sps_chapter()
    meta["SPS"] = f"data/input/john/SPS_John1.txt (sha256 {_sha(INPUT_DIR / 'SPS_John1.txt')[:16]})"
    vocab = english_vocabulary()
    out = BUILD / "translations"
    out.mkdir(parents=True, exist_ok=True)
    texts = {"WEB": web, "BSB": bsb, "OEB": oeb, "SPS": sps}
    for t, verses in texts.items():
        with (out / f"{t}.jsonl").open("w", encoding="utf-8") as f:
            for v, text in sorted(verses.items()):
                rec = {"translation": t, "book": BOOK, "chapter": CHAPTER, "verse": v, "text": text}
                if t == "SPS":
                    rec["translit"] = transliterations(text, vocab)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{t}: {len(verses)} verses — {meta[t]}")
    (out / "sources.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return texts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- source units and features

def parse_rp(path: Path) -> dict[str, list[dict]]:
    """As segment.parse_rp, for John 1."""
    verses = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["chapter"]) == CHAPTER:
                ref = f"{CODE} {row['chapter']}:{row['verse']}"
                verses[ref] = S._parsed_tokens(row["text"].split(), ref, lambda s: s)
    return verses


def parse_wh(path: Path) -> dict[str, list[dict]]:
    """As segment.parse_wh, for John 1."""
    verses: dict[str, list[str]] = {}
    ref = None
    for tok in path.read_text(encoding="utf-8").split():
        if re.fullmatch(r"\d+:\d+", tok):
            ref = f"{CODE} {tok}" if int(tok.split(":")[0]) == CHAPTER else None
            if ref:
                verses[ref] = []
        elif ref:
            verses[ref].append(tok)
    return {r: S._parsed_tokens(S._wh_text_reading(toks), r, lambda s: s.translate(S.ROBINSON_TRANSLIT))
            for r, toks in verses.items()}


def greek_units() -> list[dict]:
    """As segment.greek_units: one unit per MACULA (SBLGNT) sentence, here those of John 1."""
    root = ET.parse(SOURCES_DIR / "macula-greek/SBLGNT/lowfat/04-john.xml").getroot()
    units = []
    for sentence in root.iter("sentence"):
        words = sorted(S._words(sentence), key=lambda w: w.get(S.XML_ID))
        refs = list(dict.fromkeys(w.get("ref").split("!")[0] for w in words))
        if not all(r.startswith(f"{CODE} {CHAPTER}:") for r in refs):
            continue
        units.append(S._unit(f"{CODE}-S{len(units) + 1:03d}", "grc", refs, words, S.GRK_FIELDS,
                             S._role_map([sentence])))
    S._attach_other_editions(units, {
        "RP": parse_rp(SOURCES_DIR / "byzantine-majority-text/csv-unicode/strongs/with-parsing/JOH.csv"),
        "WH": parse_wh(SOURCES_DIR / "greektext-westcott-hort/parsed/JOH.UWH")})
    return units


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_units() -> tuple[dict, dict]:
    units = {u["unit_id"]: u for u in J._jsonl(BUILD / f"sources/{CODE}.units.jsonl")}
    feats = {f["unit_id"]: f["features"] for f in J._jsonl(BUILD / f"features/{CODE}.features.jsonl")}
    return units, feats


# --------------------------------------------------------------------------- alignment and requests

def chunks(t: str) -> list[dict]:
    """As align.translation_chunks: WEB/BSB/OEB one chunk per verse; SPS one chunk per
    paragraph, without verse numbers (ref None) — exactly as in the main study, where the SPS
    manuscript has no verse numbers. The aligner never uses the refs; they only measure its
    accuracy afterwards."""
    if t != "SPS":
        return [{"text": r["text"], "ref": f"{r['chapter']}:{r['verse']}"}
                for r in J._jsonl(BUILD / f"translations/{t}.jsonl")]
    translit = {r["verse"]: r["translit"] for r in J._jsonl(BUILD / "translations/SPS.jsonl")}
    return [{"text": " ".join(text for _, text in p), "ref": None,
             "translit": [w for v, _ in p for w in translit[v]]} for p in sps_paragraphs()]


def sps_piece_verses(pieces: list[dict]) -> list[str]:
    """Diagnostic only: the verse in which each SPS piece starts (from the inline numbers),
    so that the alignment accuracy of SPS can be measured too (not possible in the main study)."""
    out = []
    for p in sps_paragraphs():
        text, starts = "", []
        for n, v in p:
            starts.append((len(text) + (1 if text else 0), n))
            text = f"{text} {v}" if text else v
        start = 0
        bounds = [m.end() for m in A.SPLIT_RE.finditer(text)] + [len(text)]
        for end in bounds:
            seg = text[start:end]
            if seg.strip():
                at = start + len(seg) - len(seg.lstrip())
                out.append(f"{CHAPTER}:{max(n for s, n in starts if s <= at)}")
            start = end
    if len(out) != len(pieces):
        raise SystemExit("SPS piece/verse mapping out of step")
    return out


def requests(units: dict, feats: dict) -> list[dict]:
    """As judge.build_requests, for John 1."""
    reqs = []
    for t in TRANSLATIONS:
        beads = J.merge_unaligned(J._jsonl(BUILD / f"align/{t}.{CODE}.jsonl"))
        for k, bead in enumerate(beads):
            before = beads[k - 1]["english"] if k else ""
            after = beads[k + 1]["english"] if k + 1 < len(beads) else ""
            req = {"id": J._blind_id(t, CODE, bead["units"]), "translation": t, "book": CODE,
                   "units": bead["units"], "english": bead["english"], "before": before, "after": after,
                   "edition": BASE_EDITION.get(t), "unaligned_english": bead.get("unaligned", []),
                   "fids": [f["fid"] for u in bead["units"] for f in feats[u]]}
            req["prompt"] = J.render_request(req, units, feats)
            reqs.append(req)
    return reqs


def main(argv: list[str]) -> None:
    fetch_sources()
    translations()
    units = greek_units()
    write_jsonl(BUILD / f"sources/{CODE}.units.jsonl", units)
    kinds = {n: Counter(v["kind"] for u in units for v in u["variants"][n]) for n in ("RP", "WH")}
    print(f"{CODE}: {len(units)} sentence units, {sum(len(u['tokens']) for u in units)} source tokens; "
          f"differences SBLGNT<->RP {dict(kinds['RP'])}, SBLGNT<->WH {dict(kinds['WH'])}")
    counts: Counter = Counter()
    feat_rows = []
    for u in units:
        fs = F.unit_features(u)
        counts.update(x["class"] for x in fs)
        feat_rows.append({"unit_id": u["unit_id"], "features": fs})
    write_jsonl(BUILD / f"features/{CODE}.features.jsonl", feat_rows)
    print(f"{CODE}: {sum(counts.values())} information items {dict(counts.most_common())}")
    by_id = {u["unit_id"]: u for u in units}
    summary = {}
    for t in TRANSLATIONS:
        pieces = A.english_pieces(chunks(t))
        beads = A.align(units, pieces)
        write_jsonl(BUILD / f"align/{t}.{CODE}.jsonl", beads)
        summary[t] = {"pieces": len(pieces), "beads": dict(Counter(b["bead"] for b in beads).most_common())}
        if t == "SPS":  # diagnostic copy with verse refs; the alignment above is untouched
            pieces = [p | {"ref": r} for p, r in zip(pieces, sps_piece_verses(pieces))]
            beads = [b | {"refs": sorted({pieces[j]["ref"] for j in b["pieces"]})} for b in beads]
        summary[t] |= A.verse_accuracy(beads, by_id) | A.piece_accuracy(beads, pieces, by_id)
        print(t, summary[t])
    (BUILD / "align/summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    units_d, feats_d = load_units()
    reqs = requests(units_d, feats_d)
    J.OUT = BUILD / "judge"  # the judge module writes its sets, state and results here
    J.write_set("main", reqs)
    print(f"{len(reqs)} judge requests per judge "
          f"({ {t: sum(r['translation'] == t for r in reqs) for t in TRANSLATIONS} }) → {J.OUT}/sets/main.jsonl")
    validation_sets()


def validation_sets() -> None:
    """Planted errors (with their unperturbed controls) and the 10% retest, by the main study's
    own functions (validate.build_perturbations, build_retest) with its settings (plan.PER_CELL,
    the same seeds). Error types with fewer candidates in John 1 get fewer cases, the same
    number for every translation, exactly as in the main study."""
    V.OUT, V.load_sources = BUILD / "judge", load_units
    pert = V.build_perturbations(PER_CELL)
    J.write_set("perturb", pert)
    retest = V.build_retest()
    J.write_set("retest", retest)
    print(f"planted errors: {len(pert) // 2} pairs (perturbed + control); retest: {len(retest)} requests")


if __name__ == "__main__":
    main(sys.argv)
