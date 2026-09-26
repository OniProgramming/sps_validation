"""Computational validation of the judge (no human raters).

1. Known-answer (perturbation) tests. A controlled error is injected into the
   English of a request; we know which feature it damages. The perturbed
   request is judged like any other. The validator "detects" the error when the
   damaged feature is judged worse than in the unperturbed original.
   The same number of cases is drawn per translation and per error type, and
   only English words are ever perturbed, so every translation is tested alike.

       modifier_drop LEX   the English adjective rendering a source adjective is deleted
       wrong_sense   LEX   an English noun is replaced by an unrelated noun
       number_flip   NUM   an English noun after a number-neutral determiner (the, his …) is
                           switched singular ↔ plural, with correct English inflection
       negation_drop NEG   the English negator is removed (did not go → did go)
       tense_shift   ASP   an English past verb is put in the future (said → will say)
       (role_swap, ARG: implemented but unused; too few grammatical swaps in all four translations)
       addition      —     an unsupported phrase is appended (expected: an 'unsupported' addition)

   Every edit keeps the sentence grammatical, so a judge cannot detect it from
   broken English alone. Each perturbed request has an unperturbed control copy,
   judged in the same run: detection on the perturbed text is the sensitivity,
   "detection" on the identical control is the false-alarm rate.

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

from .align import STOP, skeleton, stem
from .judge import load_sources, render_request, write_set, _jsonl, OUT

PER_CELL = 40  # cases per translation × error type
SEED = 20260925

UNRELATED = ["basket", "feather", "lantern", "pebble", "thread", "ladder", "anchor", "orchard"]
ADDITIONS = [", with great joy", ", secretly at night", ", in the city of gold", ", together with their camels"]
PAST_TO_BASE = {
    "said": "say", "was": "be", "were": "be", "went": "go", "came": "come", "saw": "see",
    "took": "take", "gave": "give", "made": "make", "had": "have", "knew": "know", "bore": "bear",
    "heard": "hear", "brought": "bring", "sent": "send", "told": "tell",
    "found": "find", "ate": "eat", "rose": "rise", "sat": "sit", "spoke": "speak",
    "built": "build", "began": "begin", "became": "become", "left": "leave", "kept": "keep",
    "blessed": "bless", "died": "die", "lived": "live", "called": "call", "loved": "love",
}
DETERMINERS = {"the", "a", "an", "his", "her", "their", "your", "my", "our", "its", "thy", "this", "that",
               "these", "those", "every", "each", "all", "some", "no"}
NUMBER_NEUTRAL = {"the", "his", "her", "their", "your", "my", "our", "its", "thy", "thine"}
PREPOSITIONS = {"of", "to", "in", "on", "from", "with", "by", "at", "into", "upon", "over", "under",
                "before", "after", "among", "unto", "toward", "towards", "about", "through", "against",
                "between", "near", "behind", "beside", "around", "within", "without"}
IRREGULAR_PLURAL = {"man": "men", "woman": "women", "child": "children", "foot": "feet", "tooth": "teeth",
                    "ox": "oxen", "mouse": "mice", "goose": "geese", "brother": "brothers",
                    "wife": "wives", "life": "lives", "knife": "knives", "loaf": "loaves", "calf": "calves",
                    "leaf": "leaves", "half": "halves", "thief": "thieves", "sheaf": "sheaves", "wolf": "wolves"}
SUBORDINATORS = {"when", "if", "after", "before", "until", "till", "while", "as", "once", "whenever",
                 "because", "since", "that", "who", "whom", "which", "where", "though", "although", "unless"}
AUXILIARIES = {"had", "has", "have", "having", "was", "were", "is", "are", "am", "be", "been", "being", "did",
               "does", "do", "would", "could", "should", "might", "must", "shall", "will", "may", "can", "to"}
INVARIANT = {"sheep", "deer", "fish", "cattle", "people", "livestock", "offspring", "seed", "flock", "herd",
             # mass and abstract nouns: a plural would sound wrong, not just mean something else
             "faith", "love", "grace", "peace", "light", "darkness", "water", "bread", "wisdom", "knowledge",
             "glory", "truth", "flesh", "blood", "gold", "silver", "dust", "food", "wine", "money", "wrath",
             "evening", "morning", "heaven", "earth", "ground", "spirit", "life", "death", "strength",
             "power", "mercy", "counsel", "hope", "joy", "fear", "anger", "understanding", "favor", "favour"}
NEGATORS = re.compile(r"\b(not|never)\b\s*|n[’']t\b", re.IGNORECASE)
CONTRACTIONS = {"won’t": "will", "won't": "will", "can’t": "can", "can't": "can"}


# A noun may be edited only where nothing later in the sentence agrees with it:
# it must be followed by punctuation, the end, or a word that starts a new phrase.
SAFE_AFTER = {"and", "or", "but", "of", "in", "to", "from", "with", "for", "at", "by", "on", "into",
              "upon", "over", "under", "before", "after", "among", "unto", "toward", "towards", "through"}
IRREGULAR_PLURAL_FORMS = {"children", "men", "women", "people", "feet", "teeth", "oxen", "geese", "mice",
                          "cattle", "sheep", "deer", "fish", "brethren", "kine", "swine"}


def safe_noun_slot(text: str, s: int, e: int) -> bool:
    after = text[e:]
    if re.match(r"\s*([.,;:!?”’\"')\]—–]|$)", after):
        return True
    nxt = re.match(r"\s+([A-Za-z]+)", after)
    return bool(nxt) and nxt.group(1).lower() in SAFE_AFTER


def english_words(text: str):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z’']+", text)]


def find_rendering(text: str, tok: dict):
    """Position of an English word whose stem matches the token's English gloss."""
    glosses = {stem(w) for w in re.findall(r"[A-Za-z]+", (tok.get("english") or "") + " " + (tok.get("gloss") or ""))
               if w.lower() not in STOP and len(w) > 2}
    hits = [(s, e, w) for s, e, w in english_words(text) if stem(w) in glosses and w.lower() not in STOP]
    if not hits and tok.get("type") == "proper" and tok.get("transliteration"):
        # names spelled from the source (Avraham, Yaʿaqov) are matched by consonant skeleton
        sk = skeleton(tok["transliteration"])
        hits = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Z][\wʿʾ’-]+", text)
             if len(sk) >= 3 and skeleton(m.group(0)) == sk]
    return hits[0] if len(hits) == 1 else None  # ambiguous or absent → not usable


def _feature_index(req, units, feats):
    toks = {t["id"]: t for u in req["units"] for t in units[u]["tokens"]}
    return [(f, toks.get(f["token"], {})) for u in req["units"] for f in feats[u]]


def perturb(req: dict, kind: str, units, feats, rng: random.Random):
    """Return (new English, target fid or None, description) or None if not applicable."""
    text = req["english"]
    fs = _feature_index(req, units, feats)
    if kind == "modifier_drop":
        order = [t for u in req["units"] for t in units[u]["tokens"]]
        attributive = set()
        for i, t in enumerate(order):  # an adjective next to a noun (articles skipped) modifies it
            if t.get("class") != "adj":
                continue
            near = [x for x in order[max(0, i - 2): i] + order[i + 1: i + 3] if x.get("class") != "art"
                    and x.get("class") != "det"]
            if any(x.get("class") == "noun" for x in near[:1] + near[-1:]) or \
                    (i + 1 < len(order) and order[i + 1].get("class") == "noun") or \
                    (i > 0 and order[i - 1].get("class") == "noun"):
                attributive.add(t["id"])
        for f, tok in rng.sample(fs, len(fs)):
            if tok.get("id") not in attributive:
                continue
            if f["class"] != "LEX" or tok.get("class") != "adj" or tok.get("pos") in ("pronoun", "noun") \
                    or tok.get("type") == "substantive":
                continue
            hit = find_rendering(text, tok)
            if not hit:
                continue
            s, e, w = hit
            prev = re.findall(r"[A-Za-z’']+", text[:s])[-1:] or [""]
            if w.endswith("s") and not w.endswith(("ss", "ous")):
                continue  # matched a noun ("things"), not an adjective
            if not re.match(r"\s+[a-z]", text[e:]) or prev[0].lower() not in DETERMINERS:
                continue  # only "the/a/his great city": attributive, never predicate ("were naked")
            before = text[:s].rstrip()
            if before.endswith((" a", " an", "A", "An")) or before in ("a", "an"):
                continue  # "a good man" → "a man" is fine, but "an old man" → "an man" is not
            new = re.sub(r"\s{2,}", " ", text[:s] + text[e:]).replace(" ,", ",")
            return new, f["fid"], f"modifier_drop: '{w}'"
    if kind == "wrong_sense":
        for f, tok in rng.sample(fs, len(fs)):
            if f["class"] != "LEX" or tok.get("class") != "noun" or tok.get("type") == "proper":
                continue
            hit = find_rendering(text, tok)
            if not hit or not hit[2].islower():
                continue
            s, e, w = hit
            if w.lower() in IRREGULAR_PLURAL_FORMS or not safe_noun_slot(text, s, e):
                continue  # "The children are" → "The anchor are" would break agreement
            article = re.search(r"\b(a|an)\s+$", text[:s], re.I)
            options = [x for x in UNRELATED if x not in text and
                       (not article or (article.group(1).lower() == "an") == (x[0] in "aeiou"))]
            if not options:
                continue
            repl = rng.choice(options)
            if w.endswith("s") and not w.endswith("ss"):
                repl += "s"
            return text[:s] + repl + text[e:], f["fid"], f"wrong_sense: '{w}'→'{repl}'"
    if kind == "number_flip":
        for f, tok in rng.sample(fs, len(fs)):
            if f["class"] != "NUM":
                continue
            hit = find_rendering(text, tok)
            if not hit:
                continue
            s, e, w = hit
            prev = re.findall(r"[A-Za-z’']+", text[:s])[-2:]
            if len(prev) < 2 or prev[0].lower() not in PREPOSITIONS or prev[1].lower() not in NUMBER_NEUTRAL \
                    or not w.islower():
                continue  # only "of the sons", "to his father": a prepositional object governs no verb
            if not safe_noun_slot(text, s, e):
                continue  # "to the man who was" → "to the men who was" would break agreement
            repl = inflect_number(w)
            if repl:
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
            if hit and hit[2].lower() in PAST_TO_BASE:
                s, e, w = hit
                clause = re.split(r"[.,;:!?—–“”\"]", text[:s])[-1]  # the verb's own clause, up to the verb
                words = [x.lower() for x in re.findall(r"[A-Za-z’']+", clause)]
                if any(x in AUXILIARIES or x.endswith(("n't", "n’t")) for x in words):
                    continue  # "had (already quietly) left", "did not go", "to …": not a simple past
                if any(x in SUBORDINATORS for x in words):
                    continue  # "when his master saw" → "when … will see" is not idiomatic English
                repl = "will " + PAST_TO_BASE[w.lower()]  # grammatical with any subject
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
            has_the = [bool(re.search(r"\bthe\s+$", text[: h[0]], re.I)) for h in (h0, h1)]
            names = h0[2][0].isupper() and h1[2][0].isupper() and not any(has_the)
            the_nouns = all(has_the) and h0[2].islower() and h1[2].islower()
            if not (names or the_nouns) or h0[2].endswith("s") != h1[2].endswith("s"):
                continue  # swap only like with like, so the sentence stays grammatical
            (s0, e0, w0), (s1, e1, w1) = sorted([h0, h1])
            new = text[:s0] + w1 + text[e0:s1] + w0 + text[e1:]
            return new, f["fid"], f"role_swap: '{w0}'↔'{w1}'"
    if kind == "addition":
        m = re.search(r"[.;:,!?”’\"]*$", text)
        phrase = rng.choice(ADDITIONS)
        return text[: m.start()] + phrase + text[m.start():], None, f"addition: '{phrase.strip(', ')}'"
    return None


# role_swap (ARG) is kept in perturb() but not used: in the pilot data too few swaps could be built
# that stay grammatical in all four translations, and unequal counts would bias the comparison.
KINDS = ["modifier_drop", "wrong_sense", "number_flip", "negation_drop", "tense_shift", "addition"]


def inflect_number(word: str) -> str | None:
    """Singular ↔ plural with correct English inflection; None when unsafe."""
    if word in INVARIANT or word.endswith(("ness", "ity", "tion", "ence", "ance", "ship", "dom", "hood",
                                           "ed", "ing", "ly", "ful", "ous", "ive", "al")):
        return None  # abstract nouns, and words that are not nouns at all (participles, adjectives)
    for sg, pl in IRREGULAR_PLURAL.items():
        if word == sg:
            return pl
        if word == pl:
            return sg
    if word.endswith("ves"):
        return None  # wives/lives/… are handled by the table above; other -ves words are unsafe
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if re.search(r"[^aeiou]y$", word):
        return word[:-1] + "ies"
    if word.endswith(("ch", "sh", "ss", "x", "z")):
        return word + "es"
    if re.fullmatch(r"[a-z]{3,}", word) and not word.endswith(("f", "fe", "o", "us", "is")):
        return word + "s"
    return None


def build_perturbations(per_cell: int = PER_CELL) -> list[dict]:
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
                meta = {"kind": kind, "target": fid, "change": desc}
                q = dict(r, english=new, id=f"p{n:03d}{kind[:3]}{r['id'][1:12]}", base_id=r["id"],
                         perturbation=meta)
                q["prompt"] = render_request(q, units, feats)
                c = dict(r, id=f"c{n:03d}{kind[:3]}{r['id'][1:12]}", base_id=r["id"],
                         perturbation=dict(meta, control=True))
                out += [q, c]  # c: identical to the original text, judged in the same run
                n += 1
                if n == per_cell:
                    break
            print(t, kind, n)
    # Same number of cases per error type for every translation: trim to the smallest.
    for kind in KINDS:
        cases = {t: [q for q in out if q["translation"] == t and q["perturbation"]["kind"] == kind]
                 for t in ("WEB", "BSB", "OEB", "SPS")}
        keep = min(len(v) for v in cases.values()) // 2 * 2  # pairs (perturbed + control)
        drop = {id(q) for v in cases.values() for q in v[keep:]}
        out = [q for q in out if id(q) not in drop]
        print(f"{kind}: {keep // 2} cases per translation")
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
