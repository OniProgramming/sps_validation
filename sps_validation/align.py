"""Align source sentence units to each translation's running English text.

Verse numbers are *not* used: every translation is treated as one continuous
text per book, and the same algorithm is applied to all four. (For WEB, BSB
and OEB the verse numbers in their files are used afterwards, only to measure
how accurate the aligner is.)

Method
    1. The English text is cut into pieces at sentence and clause punctuation
       (.;:?! always; a comma only before an opening quote or a coordinating
       conjunction).
    2. Each source unit gets a bag of keys: the stemmed English glosses of its
       words (MACULA) and a consonant skeleton of each word's transliteration
       (so that "Yosef" or "reshit" in SPS matches יוֹסֵף / רֵאשִׁית).
       Each piece gets the stemmed content words and the skeletons of names and
       SPS transliterations it contains.
    3. Dynamic programming finds the monotonic segmentation that maximises
       key overlap (IDF-weighted F1) with a length prior, allowing one unit to
       take 1–8 pieces, 2–4 units to share one piece, 2:2, and 1:0 / 0:1
       (omission / addition).
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

STOP = set(
    """a an the and or but so then yet nor for of to in on at by with from into onto upon
    over under as is are was were be been being am do does did has have had will shall
    would should may might can could must not no it its he him his she her hers they them
    their theirs we us our ours you your yours i me my mine this that these those there
    here who whom whose which what when where while if also all any each every some such
    than too very up out about against between through during before after above below
    again further once only own same both few more most other off down let now just"""
    .split()
)

BEADS = [(1, b) for b in range(1, 9)] + [(a, 1) for a in range(2, 5)] + [(2, 2), (1, 0), (0, 1)]
MERGE_PENALTY = 0.05
SKIP_COST = 1.0
LENGTH_WEIGHT = 0.15

CONJ = r"(?:and|but|or|so|then|yet|nor|now|therefore|for)"
SPLIT_RE = re.compile(
    rf"(?<=[.;:?!])[”’\"')\]]*\s+|(?<=,)[”’\"]?\s+(?=[“‘\"(]|{CONJ}\b)|(?<=,[”’\"])\s+",
    re.IGNORECASE,
)

GREEK_LATIN = str.maketrans(
    {"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e", "θ": "th", "ι": "i",
     "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
     "ς": "s", "τ": "t", "υ": "y", "φ": "ph", "χ": "ch", "ψ": "ps", "ω": "o"}
)


# --------------------------------------------------------------------------- keys

def stem(word: str) -> str:
    w = word.lower().strip("’'")
    for suf in ("ingly", "edly", "ness", "ment", "ings", "ing", "ies", "ied", "est", "ers",
                "ed", "es", "er", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w


def skeleton(word: str) -> str:
    """Consonant skeleton shared by academic and popular transliterations."""
    w = unicodedata.normalize("NFD", word.lower().translate(GREEK_LATIN))
    w = re.sub(r"[^a-z]", "", w)
    w = re.sub(r"iy(?![aeiou])", "i", w)  # hireq-yod written as "iy" in academic transliteration
    for a, b in (("sh", "s"), ("ch", "k"), ("kh", "k"), ("ph", "p"), ("th", "t"),
                 ("ts", "s"), ("tz", "s"), ("ps", "ps")):
        w = w.replace(a, b)
    w = w.translate(str.maketrans("fvwcqzjx", "pbbkksyk"))
    w = re.sub(r"[aeiouy]", "", w[1:]) if w[:1] in "aeiou" else w[:1] + re.sub(r"[aeiou]", "", w[1:])
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


def english_keys(text: str) -> set[str]:
    return {f"e:{stem(w)}" for w in re.findall(r"[A-Za-z’']+", text) if w.lower() not in STOP and len(w) > 1}


def unit_keys(unit: dict) -> Counter:
    keys: Counter = Counter()
    for t in unit["tokens"]:
        for field in ("english", "gloss"):
            for k in english_keys((t.get(field) or "").replace(".", " ")):
                keys[k] = 1
        form = t.get("transliteration") or (t["text"] if unit["lang"] == "grc" else "")
        sk = skeleton(form)
        if len(sk) >= 3:
            keys[f"t:{sk}"] = 1
    return keys


def piece_keys(text: str, translit: list[str]) -> set[str]:
    keys = english_keys(text)
    names = [w for w in re.findall(r"\b[A-Z][\wʿʾ’-]+", text)] + translit
    for w in names:
        for part in re.split(r"[-\s]", w):
            sk = skeleton(part)
            if len(sk) >= 3:
                keys.add(f"t:{sk}")
    return keys


# --------------------------------------------------------------------------- English side

def english_pieces(chunks: list[dict]) -> list[dict]:
    """chunks: [{'text', 'ref' (optional), 'translit': [...]}] in reading order."""
    pieces: list[dict] = []
    for c in chunks:
        start = 0
        bounds = [m.end() for m in SPLIT_RE.finditer(c["text"])] + [len(c["text"])]
        for end in bounds:
            seg = c["text"][start:end].strip()
            start = end
            if not seg:
                continue
            tl = [t for t in c.get("translit", []) if t in seg]
            pieces.append(
                {"text": seg, "ref": c.get("ref"), "keys": piece_keys(seg, tl), "n": len(seg.split())}
            )
    return pieces


def translation_chunks(path: Path, book: str) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["book"] != book:
                continue
            if r["translation"] == "SPS":
                text = r["eval_text"]
                tl = [text[s["start"] : s["end"]] for s in r["eval_spans"] if s["type"] == "translit"]
                chunks.append({"text": text, "ref": None, "translit": tl, "para": r["para"]})
            else:
                chunks.append({"text": r["text"], "ref": f"{r['chapter']}:{r['verse']}"})
    return chunks


# --------------------------------------------------------------------------- DP

def align(units: list[dict], pieces: list[dict], band: int | None = None) -> list[dict]:
    n, m = len(units), len(pieces)
    ukeys = [unit_keys(u) for u in units]
    df: Counter = Counter()
    for k in ukeys:
        df.update(k.keys())
    idf = {k: math.log((n + 1) / (c + 0.5)) for k, c in df.items()}
    mean_idf = sum(idf.values()) / max(len(idf), 1)
    usize = [len(u["tokens"]) for u in units]
    ratio = sum(p["n"] for p in pieces) / max(sum(usize), 1)

    def score(i0, i1, j0, j1) -> float:
        S: set[str] = set()
        for i in range(i0, i1):
            S |= ukeys[i].keys()
        E: set[str] = set()
        for j in range(j0, j1):
            E |= pieces[j]["keys"]
        if not S or not E:
            return 0.0
        inter = sum(idf[k] for k in S & E)
        rec = inter / sum(idf[k] for k in S)
        prec = inter / sum(idf.get(k, mean_idf) for k in E)
        f1 = 0.0 if inter == 0 else 2 * rec * prec / (rec + prec)
        src = sum(usize[i0:i1])
        eng = sum(p["n"] for p in pieces[j0:j1])
        lp = abs(math.log((eng + 1) / (src * ratio + 1)))
        return f1 - LENGTH_WEIGHT * lp

    width = band or max(80, int(0.05 * m))
    NEG = float("-inf")
    best = [dict() for _ in range(n + 1)]
    best[0][0] = (0.0, None)

    def centre(i: int) -> int:
        return round(sum(usize[:i]) / max(sum(usize), 1) * m) if i else 0

    cum = [0]
    for s in usize:
        cum.append(cum[-1] + s)
    total = cum[-1]
    for i in range(n + 1):
        c = round(cum[i] / total * m)
        lo, hi = max(0, c - width), min(m, c + width)
        for j in range(lo, hi + 1):
            if i == 0 and j == 0:
                continue
            cand = (NEG, None)
            for a, b in BEADS:
                pi, pj = i - a, j - b
                if pi < 0 or pj < 0 or pj not in best[pi]:
                    continue
                prev = best[pi][pj][0]
                if a == 0 or b == 0:
                    gain = -SKIP_COST
                else:
                    gain = (score(pi, i, pj, j) - MERGE_PENALTY * (a + b - 2)) * (a + b) / 2
                if prev + gain > cand[0]:
                    cand = (prev + gain, (a, b))
            if cand[1] is not None:
                best[i][j] = cand
    if m not in best[n]:
        raise RuntimeError("alignment band too narrow; increase band")
    beads, i, j = [], n, m
    while i or j:
        a, b = best[i][j][1]
        beads.append((i - a, i, j - b, j))
        i, j = i - a, j - b
    beads.reverse()
    out = []
    for i0, i1, j0, j1 in beads:
        out.append(
            {
                "units": [units[i]["unit_id"] for i in range(i0, i1)],
                "pieces": list(range(j0, j1)),
                "english": " ".join(pieces[j]["text"] for j in range(j0, j1)),
                "refs": sorted({pieces[j]["ref"] for j in range(j0, j1) if pieces[j]["ref"]}),
                "bead": f"{i1 - i0}:{j1 - j0}",
                "score": round(score(i0, i1, j0, j1), 4) if i1 > i0 and j1 > j0 else None,
            }
        )
    return out


# --------------------------------------------------------------------------- evaluation of the aligner

def piece_accuracy(beads: list[dict], pieces: list[dict], units: dict[str, dict]) -> dict:
    """Share of English pieces assigned to a unit from the verse the piece belongs to."""
    hit = total = 0
    for b in beads:
        src = {r.split(" ", 1)[1] for uid in b["units"] for r in units[uid]["refs"]}
        for j in b["pieces"]:
            total += 1
            hit += pieces[j]["ref"] in src
    return {"pieces": total, "piece_in_right_verse": round(hit / max(total, 1), 4)}


def verse_accuracy(beads: list[dict], units: dict[str, dict]) -> dict:
    """Share of source units whose English lies (at least partly) in the right verse(s)."""
    hit = total = 0
    for b in beads:
        if not b["units"] or not b["refs"]:
            total += len(b["units"])
            continue
        eng = set(b["refs"])
        for uid in b["units"]:
            total += 1
            src = {r.split(" ", 1)[1] for r in units[uid]["refs"]}
            hit += bool(src & eng)
    return {"units": total, "in_right_verse": hit, "accuracy": round(hit / max(total, 1), 4)}


def main(argv: list[str]) -> None:
    only = argv[1:] or ["WEB", "BSB", "OEB", "SPS"]
    out = Path("build/align")
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for code in ("GEN", "EPH"):
        units = [json.loads(l) for l in open(f"build/sources/{code}.units.jsonl", encoding="utf-8")]
        by_id = {u["unit_id"]: u for u in units}
        book = {"GEN": "Genesis", "EPH": "Ephesians"}[code]
        for t in only:
            pieces = english_pieces(translation_chunks(Path(f"build/translations/{t}.jsonl"), book))
            beads = align(units, pieces)
            with (out / f"{t}.{code}.jsonl").open("w", encoding="utf-8") as f:
                for b in beads:
                    f.write(json.dumps(b, ensure_ascii=False) + "\n")
            kinds = Counter(b["bead"] for b in beads)
            line = {"pieces": len(pieces), "beads": dict(kinds.most_common())}
            if t != "SPS":
                line |= verse_accuracy(beads, by_id) | piece_accuracy(beads, pieces, by_id)
            summary[f"{t}.{code}"] = line
            print(t, code, line, flush=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
