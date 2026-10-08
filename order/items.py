"""ORD items: constituents placed before the verb of their clause, read from MACULA's syntax trees.

Hebrew and Greek order the constituents of a clause freely. A constituent placed before the
verb is there to give it prominence: it sets the frame or the topic of the clause, or it is in
focus (Levinsohn, Discourse Features of New Testament Greek, 2000; Runge, Discourse Grammar of
the Greek New Testament, 2010; Moshavi, Word Order in the Biblical Hebrew Finite Clause, 2010).
ORD records that placement. Like every other item it is generated mechanically, before any
translation is looked at.

Rules
- A clause is a MACULA word group of class "cl" with a verb (role v or vc).
- Each constituent of the clause standing before the verb is an ORD item if its role is
  object (o, o2), indirect object (io), predicate (p), adverbial (adv) or prepositional
  phrase (pp).

Not counted
- the subject (s): English puts the subject before the verb by grammar, so a translation can
  neither keep nor lose its position;
- vocatives, interjections and dislocated phrases (aux): they stand outside the clause;
- constituents whose place is fixed by rule: a relative or interrogative word (alone or after
  a preposition: ἐν ᾧ, לָמָּה), a negation (already a NEG item), a conjunction or particle,
  adverbial καί ("also, even", which stands before what it qualifies) and the enclitic
  adverbs ποτε, που, πως; interrogative and relative adverbs (ποῦ, πῶς, ὅπου …) and Hebrew
  טֶרֶם "not yet";
- a whole clause (participial, infinitival or subordinate, also an infinitive after a
  preposition: πρὸ τοῦ … φωνῆσαι, בְּבֹאִי) before the main verb: this is the ordinary place of
  background clauses (Levinsohn 2000, §12), not a sign of prominence;
- a Greek personal pronoun standing alone (μου, αὐτόν …; not one inside a phrase such as
  δι’ αὐτοῦ): unstressed pronouns stand before the verb by position rule, not for prominence;
- a Hebrew infinitive absolute before its finite verb (a verb form: already an ASP item);
- verbless clauses (no verb to stand before).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from sps_validation.features import GRK_NEGATION

XID = "{http://www.w3.org/XML/1998/namespace}id"
VERB = {"v", "vc"}
COUNTED = {"o", "o2", "io", "p", "adv", "pp"}
# Interrogative and relative adverbs (MACULA gives them no type) and Hebrew טֶרֶם "not yet",
# which stands before its verb by rule.
FIXED_FIRST = {"ποῦ", "πόθεν", "πῶς", "πότε", "ποσάκις", "ὅπου", "ὅθεν", "ὅτε", "ὁπότε", "ὡς",
               "καθώς", "ἕως", "טֶרֶם"}
PUNCT = " ,.;:·\u0387\u037e׃׀־—"
ROLE_NAME = {"s": "subject", "v": "verb", "vc": "verb", "o": "object", "o2": "second object",
             "io": "indirect object", "p": "predicate", "adv": "adverbial",
             "pp": "prepositional phrase", "aux": "vocative or interjection"}


def _words(node: ET.Element) -> list[ET.Element]:
    return [w for w in node.iter("w") if (w.text or "").strip()]


def _is_prep(w: ET.Element) -> bool:
    return w.get("class") == "prep" or w.get("pos") == "preposition"


def _is_function(w: ET.Element) -> bool:
    return w.get("class") in ("conj", "ptcl", "cj") or w.get("pos") in ("conjunction",)


def _place_fixed(w: ET.Element) -> bool:
    typ = (w.get("type") or "").lower()
    return ("relative" in typ or "interrogative" in typ or typ == "negative"
            or w.get("lemma") in GRK_NEGATION or w.get("lemma") in FIXED_FIRST or w.get("lemma") == "καί"
            or (w.get("class") == "adv" and typ == "indefinite"))


def counted(constituent: ET.Element) -> bool:
    if constituent.get("role") not in COUNTED or constituent.get("class") == "cl":
        return False
    if any(k.get("class") == "cl" for k in constituent):  # a clause after a preposition
        return False
    ws = _words(constituent)
    if not ws:
        return False
    lead = ws[:2] if _is_prep(ws[0]) else ws[:1]
    if any(_place_fixed(w) for w in lead):
        return False
    content = [w for w in ws if not _is_function(w) and not _is_prep(w)]
    if not content:
        return False
    if len(ws) == 1 and ws[0].get("class") == "pron" and ws[0].get("type") == "personal":
        return False
    if _is_verb(_head(ws)):
        return False
    return True


def _is_article(w: ET.Element) -> bool:
    return w.get("class") in ("det", "art") or w.get("type") == "definite article"


def _is_verb(w: ET.Element) -> bool:
    return w.get("class") == "verb" or w.get("pos") == "verb"


def _head(ws: list[ET.Element]) -> ET.Element:
    """First word that is not a preposition, article, conjunction or particle."""
    return next((w for w in ws if not (_is_function(w) or _is_prep(w) or _is_article(w))), ws[0])


def _text(ws: list[ET.Element]) -> str:
    out = ""
    for w in ws:
        after = w.get("after")
        if after is None:  # Hebrew: prefixes and suffixes are written without a space
            after = "" if w.get(XID, "").startswith("o") else " "
        out += w.text.strip() + (after + " " if after == "’" else after)
    return " ".join(out.replace("־", "־ ").split()).replace("־ ", "־").strip(PUNCT)


def unit_items(units: dict, trees: list[Path], keep: set[str] | None = None) -> dict[str, list[dict]]:
    """{unit id: ORD items}. A constituent and its verb must lie in the same unit.
    keep: only these unit ids (e.g. the Genesis sample); None means all."""
    tok_unit = {t["id"]: uid for uid, u in units.items() for t in u["tokens"]
                if keep is None or uid in keep}
    out: dict[str, list[dict]] = defaultdict(list)
    for path in trees:
        for cl in ET.parse(path).getroot().iter("wg"):
            if cl.get("class") != "cl":
                continue
            kids = [k for k in cl if k.get("role")]
            vi = next((i for i, k in enumerate(kids) if k.get("role") in VERB), None)
            if vi is None:
                continue
            verb = _words(kids[vi])
            if not verb:
                continue
            order = ", ".join(ROLE_NAME.get(k.get("role"), k.get("role")) for k in kids)
            for k in kids[:vi]:
                if not counted(k):
                    continue
                ws = _words(k)
                uid = tok_unit.get(ws[0].get(XID))
                if uid is None or tok_unit.get(verb[0].get(XID)) != uid:
                    continue
                role = ROLE_NAME[k.get("role")]
                out[uid].append({
                    "token": _head(ws).get(XID), "class": "ORD",
                    "value": f"{role} «{_text(ws)}» stands before the verb «{_text(verb)}» "
                             f"({verb[0].get(XID)}; clause order: {order})",
                    "role": k.get("role"), "words": [w.get(XID) for w in ws],
                    "verb": [w.get(XID) for w in verb], "clause": order,
                })
    for uid, fl in out.items():
        for i, f in enumerate(fl, 1):
            f["fid"] = f"{uid}/ORD{i}"
    return dict(out)
