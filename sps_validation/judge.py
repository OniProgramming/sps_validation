"""Judge each aligned source sentence (group) against one translation.

For every information feature of the source words (features.py), the judge
decides whether the English conveys it, and it lists English material that has
no source counterpart. Every translation gets exactly the same instructions;
the translation's name never appears in a request (blinding).

One request = one alignment bead (1+ source sentences and their English),
plus the English immediately before and after (so information moved across a
boundary is not scored as lost).

Two judges from different model families receive identical requests:
    claude   Anthropic, JUDGE_CLAUDE_MODEL (default claude-opus-5), Message Batches API
    gpt      OpenAI,    JUDGE_GPT_MODEL    (default gpt-5),         Batch API (/v1/responses)
Refusal fallbacks to other models are deliberately not used: every judgement
comes from the named model; refusals are recorded and reported.

Request sets (build/judge/sets/<set>.jsonl): main (validate.py adds perturb, retest).

    python -m sps_validation.judge prepare                 # write the main set
    python -m sps_validation.judge pilot <judge> N          # N random main requests, synchronously
    python -m sps_validation.judge run <judge> <set>        # batch submit → wait → collect (resumable)
Results: build/judge/results/<judge>/<set>.jsonl
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
JUDGES = {
    "claude": os.environ.get("JUDGE_CLAUDE_MODEL", "claude-opus-5"),
    "gpt": os.environ.get("JUDGE_GPT_MODEL", "gpt-5"),
}
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
                req = {"id": _blind_id(t, code, bead["units"]), "translation": t, "book": code,
                       "units": bead["units"], "english": bead["english"], "before": before,
                       "after": after, "edition": BASE_EDITION.get((t, code)),
                       "fids": [f["fid"] for u in bead["units"] for f in feats[u]]}
                req["prompt"] = render_request(req, units, feats)
                requests.append(req)
    return requests


def render_request(req: dict, units: dict, feats: dict) -> str:
    return render(
        [units[u] for u in req["units"]],
        [f for u in req["units"] for f in feats[u]],
        req["english"], req["before"], req["after"], req["edition"],
    )


def load_sources() -> tuple[dict, dict]:
    units, feats = {}, {}
    for code in BOOKS:
        units |= {u["unit_id"]: u for u in _jsonl(f"build/sources/{code}.units.jsonl")}
        feats |= {f["unit_id"]: f["features"] for f in _jsonl(f"build/features/{code}.features.jsonl")}
    return units, feats


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


# --------------------------------------------------------------------------- judges

def check_result(data: dict, fids: list[str]) -> dict:
    got = {f["fid"] for f in data["features"]}
    missing = [f for f in fids if f not in got]
    data["features"] = [f for f in data["features"] if f["fid"] in set(fids)]
    return {"status": "ok" if not missing else "incomplete", "missing": missing, **data}


class ClaudeJudge:
    name = "claude"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = JUDGES["claude"]

    def params(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        }

    def parse(self, message, fids) -> dict:
        usage = {"input": message.usage.input_tokens, "output": message.usage.output_tokens,
                 "cache_read": message.usage.cache_read_input_tokens or 0}
        if message.stop_reason == "refusal":
            return {"status": "refusal", "usage": usage}
        if message.stop_reason == "max_tokens":
            return {"status": "truncated", "usage": usage}
        text = next(b.text for b in message.content if b.type == "text")
        return check_result(json.loads(text), fids) | {"usage": usage}

    def one(self, req: dict) -> dict:
        return self.parse(self.client.messages.create(**self.params(req["prompt"])), req["fids"])

    def submit(self, reqs: list[dict]) -> list[str]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        ids = []
        for i in range(0, len(reqs), 5000):
            batch = self.client.messages.batches.create(requests=[
                Request(custom_id=r["id"], params=MessageCreateParamsNonStreaming(**self.params(r["prompt"])))
                for r in reqs[i : i + 5000]
            ])
            ids.append(batch.id)
            print("submitted", batch.id, flush=True)
        return ids

    def done(self, batch_id: str) -> bool:
        return self.client.messages.batches.retrieve(batch_id).processing_status == "ended"

    def results(self, batch_id: str, fids: dict):
        for item in self.client.messages.batches.results(batch_id):
            if item.result.type == "succeeded":
                yield item.custom_id, self.parse(item.result.message, fids[item.custom_id])
            else:
                yield item.custom_id, {"status": item.result.type}


class GptJudge:
    name = "gpt"

    def __init__(self):
        import openai

        self.client = openai.OpenAI()
        self.model = JUDGES["gpt"]

    def body(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "instructions": INSTRUCTIONS,
            "input": prompt,
            "max_output_tokens": MAX_TOKENS,
            "reasoning": {"effort": os.environ.get("JUDGE_GPT_EFFORT", "high")},
            "text": {"format": {"type": "json_schema", "name": "judgement", "schema": SCHEMA, "strict": True}},
        }

    def parse(self, resp: dict, fids) -> dict:
        usage = resp.get("usage") or {}
        usage = {"input": usage.get("input_tokens"), "output": usage.get("output_tokens")}
        if resp.get("status") == "incomplete":
            return {"status": "truncated", "usage": usage}
        for item in resp.get("output", []):
            if item.get("type") != "message":
                continue
            for c in item.get("content", []):
                if c.get("type") == "refusal":
                    return {"status": "refusal", "usage": usage}
                if c.get("type") == "output_text":
                    return check_result(json.loads(c["text"]), fids) | {"usage": usage}
        return {"status": "empty", "usage": usage}

    def one(self, req: dict) -> dict:
        resp = self.client.responses.create(**self.body(req["prompt"]))
        return self.parse(resp.model_dump(), req["fids"])

    def submit(self, reqs: list[dict]) -> list[str]:
        import io

        ids = []
        for i in range(0, len(reqs), 5000):
            lines = [json.dumps({"custom_id": r["id"], "method": "POST", "url": "/v1/responses",
                                 "body": self.body(r["prompt"])}, ensure_ascii=False)
                     for r in reqs[i : i + 5000]]
            f = self.client.files.create(file=("batch.jsonl", io.BytesIO("\n".join(lines).encode())),
                                         purpose="batch")
            batch = self.client.batches.create(input_file_id=f.id, endpoint="/v1/responses",
                                               completion_window="24h")
            ids.append(batch.id)
            print("submitted", batch.id, flush=True)
        return ids

    def done(self, batch_id: str) -> bool:
        return self.client.batches.retrieve(batch_id).status in ("completed", "failed", "expired", "cancelled")

    def results(self, batch_id: str, fids: dict):
        batch = self.client.batches.retrieve(batch_id)
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                item = json.loads(line)
                resp = (item.get("response") or {})
                if resp.get("status_code") == 200:
                    yield item["custom_id"], self.parse(resp["body"], fids[item["custom_id"]])
                else:
                    yield item["custom_id"], {"status": "errored", "error": item.get("error") or resp}


def get_judge(name: str):
    return {"claude": ClaudeJudge, "gpt": GptJudge}[name]()


def pilot(judge_name: str, n: int) -> None:
    judge = get_judge(judge_name)
    reqs = _jsonl(OUT / "sets" / "main.jsonl")
    random.Random(n).shuffle(reqs)
    out = OUT / "pilot" / judge_name
    out.mkdir(parents=True, exist_ok=True)
    for r in reqs[:n]:
        res = judge.one(r)
        (out / f"{r['id']}.json").write_text(json.dumps({"request": r, "result": res}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(r["id"], res["status"], res.get("usage"), flush=True)


def run(judge_name: str, set_name: str) -> None:
    """Submit (unless already submitted), wait, collect; retry failed items once synchronously."""
    judge = get_judge(judge_name)
    reqs = {r["id"]: r for r in _jsonl(OUT / "sets" / f"{set_name}.jsonl")}
    state_path = OUT / "state" / f"{judge_name}.{set_name}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        batch_ids = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        batch_ids = judge.submit(list(reqs.values()))
        state_path.write_text(json.dumps(batch_ids), encoding="utf-8")
    while not all(judge.done(b) for b in batch_ids):
        time.sleep(60)
    fids = {i: r["fids"] for i, r in reqs.items()}
    results = {}
    for b in batch_ids:
        for rid, res in judge.results(b, fids):
            results[rid] = res
    retry = [i for i in reqs if results.get(i, {}).get("status") not in ("ok", "refusal")]
    for rid in retry:
        try:
            results[rid] = judge.one(reqs[rid]) | {"retried": True}
        except Exception as e:  # recorded, reported as missing
            results[rid] = {"status": "failed", "error": str(e)}
    out = OUT / "results" / judge_name
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with (out / f"{set_name}.jsonl").open("w", encoding="utf-8") as f:
        for rid, r in reqs.items():
            res = results.get(rid, {"status": "missing"})
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            meta = {k: r[k] for k in ("id", "translation", "book", "units") if k in r}
            meta |= {k: r[k] for k in ("perturbation", "base_id") if k in r}
            f.write(json.dumps(meta | {"model": judge.model, "result": res}, ensure_ascii=False) + "\n")
    print(judge_name, set_name, counts)


def _jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "prepare"
    if cmd == "prepare":
        reqs = build_requests()
        (OUT / "sets").mkdir(parents=True, exist_ok=True)
        write_set("main", reqs)
        per: dict = {}
        for r in reqs:
            per[(r["translation"], r["book"])] = per.get((r["translation"], r["book"]), 0) + 1
        chars = sum(len(r["prompt"]) for r in reqs)
        print(len(reqs), "requests", per, f"~{chars / 3.2 / 1e6:.1f}M prompt tokens (rough)")
    elif cmd == "pilot":
        pilot(argv[2], int(argv[3]) if len(argv) > 3 else 5)
    elif cmd == "run":
        run(argv[2], argv[3] if len(argv) > 3 else "main")


def write_set(name: str, reqs: list[dict]) -> None:
    (OUT / "sets").mkdir(parents=True, exist_ok=True)
    with (OUT / "sets" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
