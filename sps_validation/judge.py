"""Judge each aligned source sentence (group) against one translation.

For every information feature of the source words (features.py), the judge
decides whether the English conveys it, and it lists English material that has
no source counterpart. Every translation gets exactly the same instructions;
the translation's name never appears in a request (blinding).

One request = one alignment bead (1+ source sentences and their English),
plus the English immediately before and after (so information moved across a
boundary is not scored as lost).

Usage
    python -m sps_validation.judge prepare          # write requests → build/judge/requests/
    python -m sps_validation.judge pilot N          # run N random requests synchronously
    python -m sps_validation.judge submit           # send everything as Message Batches
    python -m sps_validation.judge collect          # fetch batch results → build/judge/results/

The judge model is set by JUDGE_MODEL (default: claude-opus-5). Server-side
refusal fallbacks are deliberately *not* enabled: every judgement must come
from the named model; refusals are recorded and reported instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

TRANSLATIONS = ("WEB", "BSB", "OEB", "SPS")
BOOKS = {"GEN": "Genesis", "EPH": "Ephesians"}
BASE_EDITION = {  # the edition a translation follows where it differs from WLC/SBLGNT
    ("WEB", "GEN"): "BHS", ("WEB", "EPH"): "RP", ("OEB", "EPH"): "WH",
}
OUT = Path("build/judge")
MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5")
MAX_TOKENS = 16000

OUTCOMES = ["retained", "partial", "lost", "distorted", "not_in_base"]

INSTRUCTIONS = """You are evaluating how faithfully an English translation conveys the information of a Hebrew or Greek source text. You are one step of a fixed, published measurement procedure; apply its rules literally and identically to every input.

## Input
- SOURCE: one or more source sentences, word by word. Each word has an id, its form, lemma, parsing and a lexical gloss.
- FEATURES: the list of information features to judge. Each belongs to one source word:
  LEX lexical sense / semantic field · ASP aspect or tense-form · STEM Hebrew derived-stem meaning · VOICE middle/passive · MOOD non-indicative mood · REF who is meant (person/gender/number) · NUM noun number · DEF definiteness · REL relation expressed by a word or form (preposition, conjunction, construct/genitive/dative relation…) · NEG negation · ARG who does what to whom (semantic role).
- ENGLISH: the passage of the translation aligned to the source. BEFORE / AFTER: the neighbouring English, for reference only.
- TEXT NOTES (optional): places where this translation follows a different edition of the source.

## The single criterion
For each feature ask: **is this piece of source information conveyed by the English text, as written, to a reader of that text?**
Judge what the English says, not how it says it. Word order, word class, sentence division, and literal vs. idiomatic style are irrelevant: an idiomatic rendering that conveys the information is retained; a literal rendering that conveys a different meaning is distorted.

Outcomes:
- retained: conveyed.
- partial: conveyed only in part — narrowed, broadened, weakened, or left vaguer than the source.
- lost: not conveyed.
- distorted: the English conveys different information (wrong sense, time, agent, relation, referent…).
- not_in_base: the TEXT NOTES show that this translation's source edition lacks the word or reads it differently; the feature is not scored.

Rules that apply equally to every translation:
1. Words written in the English text in a non-English form (transliterations such as *elohim*, *nephesh*, *charis*) are judged by the same criterion as any other word: is the information conveyed to a reader of this English text? Decide from the text itself (context, established English usage). Set transliterated=true for such features.
2. Proper names: a name is retained if the person, place or people referred to is identifiable, whatever its spelling.
3. An explicit alternative in the text such as [[a|the]] or [[x / y]] presents both readings. If the source is genuinely open between them, the ambiguity is conveyed.
4. If the information is conveyed in BEFORE or AFTER instead of ENGLISH (the alignment boundary fell differently), score it as if it were in ENGLISH and set displaced=true.
5. ARG: retained when the English makes clear who does what to whom, in any construction (active, passive, nominalisation…).
6. Grammatical features the English cannot express grammatically (e.g. a Hebrew stem, a Greek middle) are retained when their meaning contribution is conveyed by other means, and lost when it is not.
7. Do not reward or penalise style, fluency, archaism or modernity.

## Additions
List English content that corresponds to no source word in SOURCE, classified as:
- grammatical: required by English grammar (articles, auxiliaries, copula, dummy subjects, pronouns resuming a known subject);
- explicitation: makes explicit something implicit in the source (e.g. names the referent of a pronoun);
- unsupported: adds information the source does not contain.
Do not list material that belongs to BEFORE/AFTER.

## Output
Return one entry for every feature id given, in the same order, and nothing else. For each: the English words that carry it (empty if none) and a short reason (max. 20 words)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fid": {"type": "string"},
                    "outcome": {"type": "string", "enum": OUTCOMES},
                    "english": {"type": "string"},
                    "transliterated": {"type": "boolean"},
                    "displaced": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["fid", "outcome", "english", "transliterated", "displaced", "reason"],
                "additionalProperties": False,
            },
        },
        "additions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "english": {"type": "string"},
                    "type": {"type": "string", "enum": ["grammatical", "explicitation", "unsupported"]},
                    "reason": {"type": "string"},
                },
                "required": ["english", "type", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["features", "additions"],
    "additionalProperties": False,
}

TOKEN_FIELDS = ("morph", "type", "stem", "person", "gender", "number", "state", "case",
                "tense", "voice", "mood")


# --------------------------------------------------------------------------- building requests

def build_requests() -> list[dict]:
    requests = []
    for code in BOOKS:
        units = {u["unit_id"]: u for u in _jsonl(f"build/sources/{code}.units.jsonl")}
        feats = {f["unit_id"]: f["features"] for f in _jsonl(f"build/features/{code}.features.jsonl")}
        for t in TRANSLATIONS:
            beads = _jsonl(f"build/align/{t}.{code}.jsonl")
            for k, bead in enumerate(beads):
                if not bead["units"]:
                    continue  # English with no source counterpart: counted as addition by report.py
                before = beads[k - 1]["english"] if k else ""
                after = beads[k + 1]["english"] if k + 1 < len(beads) else ""
                prompt = render(
                    [units[u] for u in bead["units"]],
                    [f for u in bead["units"] for f in feats[u]],
                    bead["english"], before, after,
                    BASE_EDITION.get((t, code)),
                )
                rid = _blind_id(t, code, bead["units"])
                requests.append({"id": rid, "translation": t, "book": code,
                                 "units": bead["units"], "prompt": prompt,
                                 "fids": [f["fid"] for u in bead["units"] for f in feats[u]]})
    return requests


def render(units, features, english, before, after, edition) -> str:
    lines = ["SOURCE"]
    for u in units:
        lines.append(f"[{u['unit_id']}] {u['text']}")
        for tok in u["tokens"]:
            if not tok["text"] and not tok.get("gloss"):
                continue
            parse = ", ".join(f"{k}={tok[k]}" for k in TOKEN_FIELDS if tok.get(k))
            lines.append(f"  {tok['id']} | {tok['text'] or '∅'} | {tok.get('lemma', '')} | {parse} | "
                         f"gloss: {tok.get('gloss') or tok.get('english') or ''}")
    lines.append("\nFEATURES (fid | word id | class | value)")
    for f in features:
        lines.append(f"{f['fid']} | {f['token']} | {f['class']} | {f['value']}")
    if edition:
        notes = [v for u in units for v in u.get("variants", {}).get(edition, [])]
        if notes:
            lines.append(f"\nTEXT NOTES: this translation follows a source edition that differs here:")
            for v in notes:
                if "base_letters" in v:
                    lines.append(f"- {v['ref']}: in «{v['base_context']}» the letters «{v['base_letters']}» "
                                 f"read «{v[edition.lower() + '_letters']}» (spelling; the word is the same)")
                elif v["kind"] == "transposition":
                    if v["op"] == "insert":
                        lines.append(f"- {v['ref']}: «{v[edition.lower()]}» stands in a different position (word order only)")
                elif v["kind"] in ("orthographic", "word-division"):
                    lines.append(f"- {v['ref']}: «{v['base']}» is spelled «{v[edition.lower()]}» (same word)")
                else:
                    theirs = v.get(edition.lower(), "")
                    lines.append(f"- {v['ref']}: shown text «{v['base'] or '—'}», this translation's edition «{theirs or '—'}»")
    lines += ["\nBEFORE", before or "—", "\nENGLISH", english, "\nAFTER", after or "—"]
    return "\n".join(lines)


def _blind_id(t: str, code: str, unit_ids: list[str]) -> str:
    """Opaque request id; the mapping to translations is kept only locally."""
    return "r" + hashlib.sha256(f"{t}|{code}|{'|'.join(unit_ids)}".encode()).hexdigest()[:20]


# --------------------------------------------------------------------------- API

def _params(prompt: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }


def parse_response(message, fids: list[str]) -> dict:
    if message.stop_reason == "refusal":
        return {"status": "refusal"}
    if message.stop_reason == "max_tokens":
        return {"status": "truncated"}
    text = next(b.text for b in message.content if b.type == "text")
    data = json.loads(text)
    got = [f["fid"] for f in data["features"]]
    missing = sorted(set(fids) - set(got))
    return {"status": "ok" if not missing else "incomplete", "missing": missing, **data,
            "usage": {"input": message.usage.input_tokens, "output": message.usage.output_tokens,
                      "cache_read": message.usage.cache_read_input_tokens or 0}}


def pilot(n: int) -> None:
    import anthropic

    client = anthropic.Anthropic()
    reqs = _jsonl(OUT / "requests.jsonl")
    random.Random(n).shuffle(reqs)
    out = OUT / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    for r in reqs[:n]:
        msg = client.messages.create(**_params(r["prompt"]))
        res = parse_response(msg, r["fids"])
        (out / f"{r['id']}.json").write_text(json.dumps({"request": r, "result": res}, ensure_ascii=False, indent=1))
        print(r["id"], res["status"], res.get("usage"))


def submit() -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    reqs = _jsonl(OUT / "requests.jsonl")
    batches = []
    for i in range(0, len(reqs), 10000):
        chunk = reqs[i : i + 10000]
        batch = client.messages.batches.create(
            requests=[Request(custom_id=r["id"], params=MessageCreateParamsNonStreaming(**_params(r["prompt"])))
                      for r in chunk]
        )
        batches.append(batch.id)
        print("submitted", batch.id, len(chunk))
    (OUT / f"batches.{MODEL}.json").write_text(json.dumps(batches))


def collect() -> None:
    import anthropic

    client = anthropic.Anthropic()
    reqs = {r["id"]: r for r in _jsonl(OUT / "requests.jsonl")}
    batch_ids = json.loads((OUT / f"batches.{MODEL}.json").read_text())
    for bid in batch_ids:
        while client.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(60)
    out = OUT / "results" / MODEL
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for bid in batch_ids:
            for item in client.messages.batches.results(bid):
                r = reqs[item.custom_id]
                if item.result.type == "succeeded":
                    res = parse_response(item.result.message, r["fids"])
                else:
                    res = {"status": item.result.type}
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                f.write(json.dumps({"id": r["id"], "translation": r["translation"], "book": r["book"],
                                    "units": r["units"], "result": res}, ensure_ascii=False) + "\n")
    print(counts)


def _jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "prepare"
    if cmd == "prepare":
        reqs = build_requests()
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "requests.jsonl").open("w", encoding="utf-8") as f:
            for r in reqs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per = {}
        for r in reqs:
            per[(r["translation"], r["book"])] = per.get((r["translation"], r["book"]), 0) + 1
        chars = sum(len(r["prompt"]) for r in reqs)
        print(len(reqs), "requests", per, f"~{chars / 3.2 / 1e6:.1f}M prompt tokens (rough)")
    elif cmd == "pilot":
        pilot(int(argv[2]) if len(argv) > 2 else 5)
    elif cmd == "submit":
        submit()
    elif cmd == "collect":
        collect()


if __name__ == "__main__":
    main(sys.argv)
