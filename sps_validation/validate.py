"""Computational validation of the judge (no human raters).

1. Known-answer (perturbation) tests. A controlled error is injected into the
   English of a request; we know which feature it damages. The perturbed
   request is judged like any other. The validator "detects" the error when the
   damaged feature is judged worse than in the unperturbed original.
   The same number of cases is drawn per translation and per error type, and
   only English words are ever perturbed, so every translation is tested alike.

       delete_word   LEX   the English word rendering a content word is deleted
       wrong_sense   LEX   that word is replaced by an unrelated word
       number_flip   NUM   an English noun is switched singular ↔ plural
       negation_drop NEG   the English negator is removed
       tense_shift   ASP   a past-tense English verb is put in the present
       role_swap     ARG   the English words for agent (A0) and patient (A1) swap places
       addition      —     an unsupported phrase is appended (expected: an 'unsupported' addition)

2. Test–retest: a random 10% of the main requests is judged a second time.

3. Rule cross-checks (report.py): NEG and NUM features are also tested by a
   deterministic rule on the aligned English, and agreement with the judges is reported.

    python -m sps_validation.validate prepare     # writes sets 'perturb' and 'retest'
"""

from __future__ import annotations

import json
import random
import re
import sys

from .align import STOP, stem
from .judge import load_sources, render_request, write_set, _jsonl, OUT

PER_CELL = 40  # cases per translation × error type
SEED = 20260925

UNRELATED = ["basket", "feather", "lantern", "anchor", "harvest", "pebble", "thread", "ladder"]
ADDITIONS = [", with great joy", ", secretly at night", ", in the city of gold", ", together with their camels"]
PAST_TO_PRESENT = {
    "said": "says", "was": "is", "were": "are", "went": "goes", "came": "comes", "saw": "sees",
    "took": "takes", "gave": "gives", "made": "makes", "had": "has", "knew": "knows", "bore": "bears",
    "heard": "hears", "brought": "brings", "sent": "sends", "told": "tells",
    "found": "finds", "ate": "eats", "rose": "rises", "sat": "sits", "spoke": "speaks",
    "built": "builds", "began": "begins", "became": "becomes", "left": "leaves", "kept": "keeps",
    "blessed": "blesses", "died": "dies", "lived": "lives", "called": "calls", "loved": "loves",
}
NEGATORS = re.compile(r"\b(not|never)\b\s*|n[’']t\b", re.IGNORECASE)
CONTRACTIONS = {"won’t": "will", "won't": "will", "can’t": "can", "can't": "can"}


def english_words(text: str):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z’']+", text)]


def find_rendering(text: str, tok: dict):
    """Position of an English word whose stem matches the token's English gloss."""
    glosses = {stem(w) for w in re.findall(r"[A-Za-z]+", (tok.get("english") or "") + " " + (tok.get("gloss") or ""))
               if w.lower() not in STOP and len(w) > 2}
    hits = [(s, e, w) for s, e, w in english_words(text) if stem(w) in glosses and w.lower() not in STOP]
    return hits[0] if len(hits) == 1 else None  # ambiguous or absent → not usable


def _feature_index(req, units, feats):
    toks = {t["id"]: t for u in req["units"] for t in units[u]["tokens"]}
    return [(f, toks.get(f["token"], {})) for u in req["units"] for f in feats[u]]


def perturb(req: dict, kind: str, units, feats, rng: random.Random):
    """Return (new English, target fid or None, description) or None if not applicable."""
    text = req["english"]
    fs = _feature_index(req, units, feats)
    if kind in ("delete_word", "wrong_sense"):
        for f, tok in rng.sample(fs, len(fs)):
            if f["class"] != "LEX" or tok.get("class") not in ("noun", "verb", "adj"):
                continue
            hit = find_rendering(text, tok)
            if not hit:
                continue
            s, e, w = hit
            if kind == "delete_word":
                new = (text[:s] + text[e:]).replace("  ", " ")
            else:
                repl = next(x for x in rng.sample(UNRELATED, len(UNRELATED)) if x not in text)
                new = text[:s] + repl + text[e:]
            return new, f["fid"], f"{kind}: '{w}'"
    if kind == "number_flip":
        for f, tok in rng.sample(fs, len(fs)):
            if f["class"] != "NUM":
                continue
            hit = find_rendering(text, tok)
            if not hit:
                continue
            s, e, w = hit
            if w.endswith("s") and not w.endswith("ss"):
                repl = w[:-2] if w.endswith("es") and w[:-2].endswith(("sh", "ch", "x")) else w[:-1]
            elif re.fullmatch(r"[a-z]+", w) and not w.endswith(("s", "y")):
                repl = w + "s"
            else:
                continue
            return text[:s] + repl + text[e:], f["fid"], f"number_flip: '{w}'→'{repl}'"
    if kind == "negation_drop":
        neg = [f for f, _ in fs if f["class"] == "NEG"]
        if len(neg) == 1:
            for c, plain in CONTRACTIONS.items():
                if c in text:
                    return text.replace(c, plain, 1), neg[0]["fid"], f"negation_drop: '{c}'"
            m = NEGATORS.search(text)
            if m:
                return text[: m.start()] + text[m.end():], neg[0]["fid"], f"negation_drop: '{m.group(0).strip()}'"
    if kind == "tense_shift":
        for f, tok in rng.sample(fs, len(fs)):
            if f["class"] != "ASP" or tok.get("class") != "verb":
                continue
            if tok.get("type") not in ("qatal", "wayyiqtol") and tok.get("tense") not in ("aorist", "imperfect"):
                continue  # only source past forms rendered by an English past
            hit = find_rendering(text, tok)
            if hit and hit[2].lower() in PAST_TO_PRESENT:
                s, e, w = hit
                repl = PAST_TO_PRESENT[w.lower()]
                repl = repl.capitalize() if w[0].isupper() else repl
                return text[:s] + repl + text[e:], f["fid"], f"tense_shift: '{w}'→'{repl}'"
    if kind == "role_swap":
        toks = {t["id"]: t for u in req["units"] for t in units[u]["tokens"]}
        for f, verb in fs:
            if f["class"] != "ARG" or not f["value"].startswith("A0"):
                continue
            a1 = next((g for g, _ in fs if g["class"] == "ARG" and g["token"] == f["token"]
                       and g["value"].startswith("A1")), None)
            if not a1 or len(f["args"]) != 1 or len(a1["args"]) != 1:
                continue
            h0 = find_rendering(text, toks.get(f["args"][0], {}))
            h1 = find_rendering(text, toks.get(a1["args"][0], {}))
            if not h0 or not h1 or h0[0] == h1[0]:
                continue
            (s0, e0, w0), (s1, e1, w1) = sorted([h0, h1])
            new = text[:s0] + w1 + text[e0:s1] + w0 + text[e1:]
            return new, f["fid"], f"role_swap: '{w0}'↔'{w1}'"
    if kind == "addition":
        m = re.search(r"[.;:,!?”’\"]*$", text)
        phrase = rng.choice(ADDITIONS)
        return text[: m.start()] + phrase + text[m.start():], None, f"addition: '{phrase.strip(', ')}'"
    return None


KINDS = ["delete_word", "wrong_sense", "number_flip", "negation_drop", "tense_shift", "role_swap", "addition"]


def build_perturbations() -> list[dict]:
    units, feats = load_sources()
    main = _jsonl(OUT / "sets" / "main.jsonl")
    rng = random.Random(SEED)
    out = []
    for t in ("WEB", "BSB", "OEB", "SPS"):
        pool = [r for r in main if r["translation"] == t]
        for kind in KINDS:
            rng.shuffle(pool)
            n = 0
            for r in pool:
                p = perturb(r, kind, units, feats, rng)
                if not p:
                    continue
                new, fid, desc = p
                q = dict(r, english=new, id=f"p{n:03d}{kind[:3]}{r['id'][1:12]}", base_id=r["id"],
                         perturbation={"kind": kind, "target": fid, "change": desc})
                q["prompt"] = render_request(q, units, feats)
                out.append(q)
                n += 1
                if n == PER_CELL:
                    break
            print(t, kind, n)
    return out


def build_retest() -> list[dict]:
    main = _jsonl(OUT / "sets" / "main.jsonl")
    rng = random.Random(SEED + 1)
    by_t: dict[str, list] = {}
    for r in main:
        by_t.setdefault(r["translation"], []).append(r)
    sample = []
    for t, rs in by_t.items():
        sample += rng.sample(rs, round(len(rs) * 0.10))
    return sample


def main(argv: list[str]) -> None:
    if (argv[1] if len(argv) > 1 else "prepare") == "prepare":
        p = build_perturbations()
        write_set("perturb", p)
        r = build_retest()
        write_set("retest", r)
        print(f"perturb: {len(p)} requests; retest: {len(r)} requests")


if __name__ == "__main__":
    main(sys.argv)
