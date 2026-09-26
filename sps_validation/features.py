"""Information-feature inventory: what each source word carries.

Generated mechanically from the source annotation, before any translation is
looked at, so the judge never decides what counts as information. Every
feature is later scored retained / partial / lost / distorted against each
translation, and all features weigh the same.

Classes
    LEX    lexical sense of a content word (sense/semantic domain attached)
    ASP    Hebrew verb conjugation / Greek tense-form
    STEM   Hebrew derived stem (qal is unmarked and carries no extra feature)
    VOICE  Greek middle or passive (active is unmarked)
    MOOD   Greek non-indicative mood
    REF    person / number (/ gender) of a finite verb or pronoun: who is meant
    NUM    number of a common noun
    DEF    definiteness (article)
    REL    relation expressed by a word or form: preposition, conjunction,
           relative, construct state, genitive or dative case, directional he
    NEG    negation
    ARG    semantic role in a verb's frame (A0 agent, A1 patient, …): who does what to whom

Carries no feature of its own (its information is represented elsewhere):
grammatical agreement (adjective number/gender, gender of inanimate nouns),
the number of plurals without count meaning (NO_COUNT_PLURALS),
the Hebrew object marker אֵת (→ ARG), paragogic nun.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HEB_FINITE = {"qatal", "yiqtol", "wayyiqtol", "weqatal", "imperative", "jussive", "cohortative"}
HEB_UNMARKED_STEMS = {"qal", None}
GRK_NEGATION = {"οὐ", "μή", "οὐδέ", "μηδέ", "οὐκέτι", "μηκέτι", "οὔτε", "μήτε", "οὐχί"}
PERSON = {"first": "1", "second": "2", "third": "3"}
GENDER = {"masculine": "m", "feminine": "f", "both": "c", "common": "c", "neuter": "n"}
NUMBER = {"singular": "s", "plural": "p", "dual": "d"}


def hebrew_features(tok: dict) -> list[tuple[str, str]]:
    cls, pos, typ, morph = tok.get("class"), tok.get("pos"), tok.get("type"), tok.get("morph") or ""
    feats: list[tuple[str, str]] = []
    lexical = pos in ("noun", "verb", "adjective", "adverb") or cls in ("noun", "verb", "adj", "num")
    if cls == "pron" and pos == "pronoun" or (cls == "pron" and typ == "interrogative"):
        lexical = False
        feats.append(("REF" if typ == "personal" else "LEX", _lex_value(tok)))
    elif cls == "adj" and pos == "pronoun":  # demonstratives
        feats.append(("LEX", _lex_value(tok)))
    elif lexical and cls not in ("om",):
        feats.append(("LEX", _lex_value(tok)))
    if pos == "verb":
        feats.append(("ASP", typ or morph))
        if tok.get("stem") not in HEB_UNMARKED_STEMS and not str(tok.get("stem")).startswith("unknown"):
            feats.append(("STEM", tok["stem"]))
        if typ in HEB_FINITE or (typ or "").startswith("participle"):
            feats.append(("REF", _pgn(tok)))
    if pos == "noun" and typ == "common":
        feats.append(("NUM", tok.get("number", "")))
    if tok.get("state") == "construct" and pos in ("noun", "adjective"):
        feats.append(("REL", "construct (bound to the following noun)"))
    if cls == "art":
        feats.append(("DEF", "definite article"))
    if pos == "suffix" and cls == "pron":
        feats.append(("REF", f"pronominal suffix {_pgn(tok)}"))
    if pos == "suffix" and morph == "Sd":
        feats.append(("REL", "directional he (toward)"))
    if cls in ("prep", "cj", "rel") or (pos == "preposition" and cls != "om"):
        feats.append(("REL", _lex_value(tok)))
    if cls == "cj" and pos == "particle":  # affirmation particles (אַף, גַּם …)
        pass
    if typ == "negative":
        feats.append(("NEG", _lex_value(tok)))
    elif cls in ("ij", "ptcl") or typ in ("affirmation", "demonstrative") and pos == "particle":
        feats.append(("REL", _lex_value(tok)))
    return _dedupe(feats)


def greek_features(tok: dict) -> list[tuple[str, str]]:
    cls, typ = tok.get("class"), tok.get("type")
    feats: list[tuple[str, str]] = []
    if tok.get("lemma") in GRK_NEGATION:
        feats.append(("NEG", _lex_value(tok)))
    elif cls in ("noun", "verb", "adj", "adv", "num"):
        feats.append(("LEX", _lex_value(tok)))
    elif cls == "pron":
        if typ == "personal":
            feats.append(("REF", _pgn(tok)))
        elif typ == "relative":
            feats.append(("REL", _lex_value(tok)))
            feats.append(("REF", _pgn(tok)))
        else:
            feats.append(("LEX", _lex_value(tok)))
    if cls == "verb":
        feats.append(("ASP", tok.get("tense", "")))
        if tok.get("voice") not in (None, "active"):
            feats.append(("VOICE", tok["voice"]))
        if tok.get("mood") not in (None, "indicative"):
            feats.append(("MOOD", tok["mood"]))
        if tok.get("person"):
            feats.append(("REF", _pgn(tok)))
    if cls == "noun" and typ != "proper":
        feats.append(("NUM", tok.get("number", "")))
    if cls in ("noun", "pron") and tok.get("case") in ("genitive", "dative"):
        feats.append(("REL", f"{tok['case']} case"))
    if cls == "det":
        feats.append(("DEF", "article"))
    if cls in ("prep", "conj", "ptcl"):
        feats.append(("REL", _lex_value(tok)))
    return _dedupe(feats)


def frame_features(tokens: list[dict], lang: str) -> list[dict]:
    """ARG features from verb frames whose arguments lie inside the same unit."""
    by_id = {t["id"]: t for t in tokens}
    prefix = "o" if lang == "hbo" else ""
    out = []
    for t in tokens:
        for role, ids in re.findall(r"(A\d|AA)\s*:\s*([^A]+?)(?=\s*A\d\s*:|\s*AA\s*:|$)", t.get("frame", "")):
            args = [by_id.get(prefix + i.strip()) for i in re.split(r"[;\s]+", ids) if i.strip()]
            args = [a for a in args if a]
            if args:
                out.append(
                    {
                        "token": t["id"],
                        "class": "ARG",
                        "value": f"{role} of {t.get('lemma') or t['text']} ({t['text']}) = "
                        + " + ".join(f"{a.get('lemma') or a['text']} ({a['text']})" for a in args),
                        "args": [a["id"] for a in args],
                    }
                )
    return out


def _lex_value(tok: dict) -> str:
    # No English gloss: MACULA's glosses come from the Berean Interlinear (Greek) and
    # Cherith (Hebrew); one of the evaluated translations is the BSB, so glosses must
    # not reach the judges. The judges get the lemma, Strong's number and domain codes.
    strong = tok.get("strong") or tok.get("strongnumberx")
    parts = [tok.get("lemma", ""), f"Strong {strong}" if strong else ""]
    if tok.get("ln"):
        parts.append(f"LN {tok['ln']}")
    if tok.get("lexdomain"):
        parts.append(f"SDBH {tok['lexdomain']}")
    return " | ".join(p for p in parts if p)


def _pgn(tok: dict) -> str:
    out = "".join(
        m.get(tok.get(k) or "", "")
        for k, m in (("person", PERSON), ("gender", GENDER), ("number", NUMBER))
    )
    # Greek personal pronouns carry person only in the morph code (P-2DP)
    m = re.match(r"^[PRF]-(\d)", tok.get("morph") or "")
    if m and not tok.get("person"):
        out = m.group(1) + out
    return out


# Plural forms without count meaning (plurals of extension, abstraction or
# majesty; GKC §124, Joüon–Muraoka §136). Their grammatical number is not
# counted as information. אֱלֹהִים is exempt only when it denotes God (glossed
# "God"), not "gods".
NO_COUNT_PLURALS = {"שָׁמַיִם", "מַיִם", "פָּנֶה", "פָּנִים", "חַיִּים", "זְקֻנִים", "נְעוּרִים",
                    "מְגוּרִים", "תְּרָפִים"}


def carries_number(tok: dict) -> bool:
    lemma = tok.get("lemma")
    if lemma in NO_COUNT_PLURALS:
        return False
    if lemma == "אֱלֹהִים":  # the gloss decides God vs gods here only; it is not shown to the judges
        return (tok.get("gloss") or tok.get("english") or "").lower().startswith("gods")
    return True


def _dedupe(feats):
    return list(dict.fromkeys(feats))


def unit_features(unit: dict) -> list[dict]:
    fn = hebrew_features if unit["lang"] == "hbo" else greek_features
    out = []
    for t in unit["tokens"]:
        for cls, value in fn(t):
            if cls == "NUM" and not carries_number(t):
                continue
            out.append({"token": t["id"], "class": cls, "value": value})
    out += frame_features(unit["tokens"], unit["lang"])
    for i, f in enumerate(out, 1):
        f["fid"] = f"{unit['unit_id']}/{i}"
    return out


def main(argv: list[str]) -> None:
    out = Path("build/features")
    out.mkdir(parents=True, exist_ok=True)
    for code in ("GEN", "EPH"):
        counts: Counter = Counter()
        with open(f"build/sources/{code}.units.jsonl", encoding="utf-8") as f:
            units = [json.loads(line) for line in f]
        with (out / f"{code}.features.jsonl").open("w", encoding="utf-8") as g:
            for u in units:
                feats = unit_features(u)
                counts.update(x["class"] for x in feats)
                g.write(json.dumps({"unit_id": u["unit_id"], "features": feats}, ensure_ascii=False) + "\n")
        print(code, sum(counts.values()), "features", dict(counts.most_common()))


if __name__ == "__main__":
    main(sys.argv)
