"""Judge each aligned source sentence (group) against one translation.

For every information feature of the source words (features.py), the judge
decides whether the English conveys it, and it lists English material that has
no source counterpart. Every translation gets exactly the same instructions;
the translation's name never appears in a request (blinding).

One request = one alignment bead (1+ source sentences and their English),
plus the English immediately before and after (so information moved across a
boundary is not scored as lost).

Two judges from different model families receive identical requests:
    claude   Anthropic, JUDGE_CLAUDE_MODEL (default claude-haiku-4-5), Message Batches API
    gpt      OpenAI,    JUDGE_GPT_MODEL    (default gpt-5-mini),       Batch API (/v1/responses)
Refusal fallbacks to other models are deliberately not used: every judgement
comes from the named model; refusals are recorded and reported.

Request sets (build/judge/sets/<set>.jsonl): all (every alignment group);
plan.py derives main (the sample), retest and perturb from it.

    python -m sps_validation.judge prepare                 # write the 'all' set
    python -m sps_validation.judge pilot <judge> N          # N random main requests, synchronously
    python -m sps_validation.judge pilot-report             # readable HTML of the pilot, both judges
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
    "claude": os.environ.get("JUDGE_CLAUDE_MODEL", "claude-haiku-4-5"),
    "gpt": os.environ.get("JUDGE_GPT_MODEL", "gpt-5-mini"),
}
MAX_TOKENS = 16000

OUTCOMES = ["retained", "partial", "lost", "distorted", "not_in_base"]

INSTRUCTIONS = """You are evaluating how faithfully an English translation renders the information of a Hebrew or Greek source text, along two dimensions. You are one step of a fixed, published measurement procedure; apply its rules literally and identically to every input.

## Input
- SOURCE: one or more source sentences, word by word. Each word has an id, its form, its lemma, its parsing and its Strong's number. No English gloss is given: determine the meaning of each word yourself from the Hebrew or Greek.
- FEATURES: the list of information features to judge. Each belongs to one source word:
  LEX lexical sense / semantic field · ASP aspect or tense-form · STEM Hebrew derived-stem meaning · VOICE middle/passive · MOOD non-indicative mood · REF who is meant (person/gender/number) · NUM noun number · DEF definiteness · REL relation expressed by a word or form (preposition, conjunction, construct/genitive/dative relation…) · NEG negation · ARG who does what to whom (semantic role).
- ENGLISH: the passage of the translation aligned to the source. BEFORE / AFTER: the neighbouring English, for reference only.
- TEXT NOTES (optional): places where this translation follows a different edition of the source.

## Two dimensions, two verdicts per feature
Every feature receives two independent verdicts. Apply both to every translation in exactly the same way.

**outcome — SENSE CONVEYED.** Is this piece of source information conveyed by the English text, as written, to an ordinary reader of that text?
Judge what the English says, not how it says it. Word order, word class, sentence division, and literal vs. idiomatic style are irrelevant: an idiomatic rendering that conveys the information is retained; a literal rendering that conveys a different meaning is distorted.

**source_outcome — SOURCE PRESERVED.** Is this piece of source information preserved in the English text, so that it can be recovered from the text as written — through the English words, literal renderings, word order, transliterated source words or explicit markers — even if an ordinary reader would need effort or knowledge of the source to recover it?
Here what counts is what the text keeps of the source itself: the identity and literal sense of each source word, its grammatical distinctions and its relations. A rendering that conveys the meaning while replacing the source's own words, image or distinction preserves the meaning but not the replaced element.

Outcomes (the same scale for both verdicts):
- retained: conveyed (outcome) / preserved (source_outcome).
- partial: only in part — narrowed, broadened, weakened, vaguer, or preserved only indirectly.
- lost: not conveyed / not preserved.
- distorted: the English conveys different information (wrong sense, time, agent, relation, referent…).
- not_in_base: the TEXT NOTES show that this translation's source edition lacks the word or reads it differently; the feature is not scored (use it for both verdicts).

Rules for both dimensions, applied equally to every translation:
1. Proper names: retained if the person, place or people referred to is identifiable, whatever the spelling (Hushim / Hashum, Yosef / Joseph). Distorted only if it points to a different referent.
2. An explicit alternative in the text such as [[a|the]] or [[x / y]] presents both readings; if the source is genuinely open between them, the ambiguity is retained.
3. If the information is found in BEFORE or AFTER instead of ENGLISH (the alignment boundary fell differently), score it as if it were in ENGLISH and set displaced=true.
4. Words written in a non-English form (transliterations such as *elohim*, *nephesh*, *charis*): set transliterated=true. For SENSE, judge whether the meaning reaches an ordinary reader of this English text (context, established English usage). For SOURCE, a transliteration preserves the identity of the source word, and grammatical information visible in its form (e.g. a Greek case ending) is preserved.
5. Do not reward or penalise style, fluency, archaism or modernity.

Rules for SENSE CONVEYED (outcome):
6. ARG: retained when the English makes clear who does what to whom, in any construction (active, passive, nominalisation…).
7. Grammatical features the English cannot express grammatically are retained when their meaning contribution is conveyed by other means, and lost when it is not. STEM and VOICE: retained when the English verb or construction has the meaning the verb has in that stem or voice; a stem that is simply the verb's ordinary form needs no additional English marker.
8. REF: retained when the English makes clear who is meant. Distinctions English cannot mark (masculine vs feminine "you", singular vs plural "you", the gender of "they") are not required when the referent is clear.
9. NUM: retained when the English conveys the same count. An English plural-form noun naming the same thing (wages, clothes) and an English collective singular count as the same count. Where no count is at issue (e.g. a noun used adverbially), NUM is retained unless the English asserts a different count.
10. Fixed expressions: in expressions whose words no longer carry their separate literal sense (compound prepositions such as לִפְנֵי "before", fixed time or place expressions), a rendering that conveys the whole expression retains the features of its parts. Where the source uses a live image or metaphor, the image is part of the information.
11. Conjunctions and other REL features are retained when the relation they express (addition, sequence, contrast, cause, purpose, condition…) is conveyed, whether by a word, by clause order, or by sentence structure.

Rules for SOURCE PRESERVED (source_outcome):
12. The question is whether the text keeps a direct counterpart of the source item, not whether it shows the Hebrew or Greek form. A standard English equivalent that directly renders the source word or form preserves it: "God" for אֵל, "went" (simple past) for a narrative wayyiqtol, "his" for a 3ms suffix, "and" for וְ, "of" for a construct relation. A transliteration preserves it too.
13. retained: the item has its own direct counterpart in the text. partial: the item survives only indirectly — merged into another word or phrase, turned into a different word class or construction, its literal sense or image replaced by the meaning of a larger expression (e.g. "before" for לִפְנֵי "to the face of"), or recoverable only from context. lost: no counterpart in the text. distorted: the counterpart says something different.
14. Distinctions English grammar cannot mark at all (the gender of "you" and "they", Hebrew conjugation classes, the Greek middle when English has no corresponding form) are retained when the text uses the English category that directly corresponds (e.g. "they" for 3mp, a past tense for a narrative past); they are not penalised for what English cannot express.

## Additions
List English content that corresponds to no source word in SOURCE, classified as:
- grammatical: required by English grammar (articles, auxiliaries, copula, dummy subjects, pronouns resuming a known subject);
- explicitation: makes explicit something implicit in the source (e.g. names the referent of a pronoun);
- unsupported: adds information the source does not contain.
Do not list material that belongs to BEFORE/AFTER.

## Output
Return one entry for every feature id given, in the same order, and nothing else. For each: both verdicts, the English words that carry the feature (empty if none) and one short reason covering both verdicts (max. 15 words)."""

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
                    "source_outcome": {"type": "string", "enum": OUTCOMES},
                    "english": {"type": "string"},
                    "transliterated": {"type": "boolean"},
                    "displaced": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["fid", "outcome", "source_outcome", "english", "transliterated", "displaced", "reason"],
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
            beads = merge_unaligned(_jsonl(f"build/align/{t}.{code}.jsonl"))
            for k, bead in enumerate(beads):
                before = beads[k - 1]["english"] if k else ""
                after = beads[k + 1]["english"] if k + 1 < len(beads) else ""
                req = {"id": _blind_id(t, code, bead["units"]), "translation": t, "book": code,
                       "units": bead["units"], "english": bead["english"], "before": before,
                       "after": after, "edition": BASE_EDITION.get((t, code)),
                       "unaligned_english": bead.get("unaligned", []),
                       "fids": [f["fid"] for u in bead["units"] for f in feats[u]]}
                req["prompt"] = render_request(req, units, feats)
                requests.append(req)
    return requests


def merge_unaligned(beads: list[dict]) -> list[dict]:
    """English with no source counterpart (0:1 alignment) is appended to the preceding
    group (or prepended to the first one), so the judge sees it and can list it as an
    addition. Nothing drops out of the evaluation; the merged text is recorded."""
    out: list[dict] = []
    pending: list[str] = []
    for b in beads:
        if b["units"]:
            b = dict(b, unaligned=list(pending))
            if pending:
                b["english"] = " ".join(pending + [b["english"]])
            pending = []
            out.append(b)
        elif out:
            out[-1]["english"] += " " + b["english"]
            out[-1]["unaligned"] = out[-1].get("unaligned", []) + [b["english"]]
        else:
            pending.append(b["english"])
    return out


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
            if not tok["text"] and not tok.get("lemma"):
                continue
            parse = ", ".join(f"{k}={tok[k]}" for k in TOKEN_FIELDS if tok.get(k))
            strong = tok.get("strong") or tok.get("strongnumberx") or ""
            lines.append(f"  {tok['id']} | {tok['text'] or '∅'} | {tok.get('lemma', '')} | {parse} | Strong {strong}")
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
                 "cache_read": message.usage.cache_read_input_tokens or 0, "served_model": message.model}
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
            "reasoning": {"effort": os.environ.get("JUDGE_GPT_EFFORT", "medium")},
            "text": {"format": {"type": "json_schema", "name": "judgement", "schema": SCHEMA, "strict": True}},
        }

    def parse(self, resp: dict, fids) -> dict:
        usage = resp.get("usage") or {}
        usage = {"input": usage.get("input_tokens"), "output": usage.get("output_tokens"),
                 "served_model": resp.get("model")}
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


def fingerprint(judge, reqs: dict) -> str:
    """Identity of an experiment: model, provider settings, instructions, schema and the
    exact requests. A saved run is resumed only if all of these are unchanged."""
    probe = judge.params("") if hasattr(judge, "params") else judge.body("")
    h = hashlib.sha256()
    h.update(json.dumps({"model": judge.model, "settings": probe, "instructions": INSTRUCTIONS,
                         "schema": SCHEMA}, sort_keys=True, ensure_ascii=False).encode())
    for rid in sorted(reqs):
        h.update(rid.encode())
        h.update(reqs[rid]["prompt"].encode())
    return h.hexdigest()


def pilot(judge_name: str, n: int) -> None:
    judge = get_judge(judge_name)
    reqs = _jsonl(OUT / "sets" / "all.jsonl")
    random.Random(n).shuffle(reqs)
    out = OUT / "pilot" / judge_name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):  # a pilot reflects the current model only
        old.unlink()
    for r in reqs[:n]:
        res = judge.one(r)
        (out / f"{r['id']}.json").write_text(json.dumps({"request": r, "model": judge.model, "result": res},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
        print(r["id"], res["status"], res.get("usage"), flush=True)


def pilot_report() -> Path:
    """Readable side-by-side view of the pilot judgements (both judges)."""
    import html

    root = OUT / "pilot"
    judged = {j: {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (root / j).glob("*.json")}
              for j in JUDGES if (root / j).exists()}
    ids = sorted(set().union(*[set(v) for v in judged.values()]))
    units, feats = load_sources()
    by_fid = {f["fid"]: f for fl in feats.values() for f in fl}
    toks = {t["id"]: t for u in units.values() for t in u["tokens"]}
    agree = total = agree_sense = agree_source = 0
    colour = {"retained": "#1b7f3b", "partial": "#a36b00", "lost": "#b3261e", "distorted": "#7b1fa2",
              "not_in_base": "#666"}
    rows = []
    for rid in ids:
        any_rec = next(v[rid] for v in judged.values() if rid in v)
        req = any_rec["request"]
        out = {j: {f["fid"]: f for f in v[rid]["result"].get("features", [])} for j, v in judged.items() if rid in v}
        adds = {j: v[rid]["result"].get("additions", []) for j, v in judged.items() if rid in v}
        rows.append(f"<section><h2>{html.escape(req['translation'])} · {html.escape(', '.join(req['units']))}</h2>")
        rows.append("<p class=src>" + " ".join(html.escape(units[u]["text"]) for u in req["units"]) + "</p>")
        rows.append(f"<p class=eng>{html.escape(req['english'])}</p>")
        rows.append("<table><tr><th>word</th><th>information</th>" +
                    "".join(f"<th>{j}</th>" for j in out) + "</tr>")
        for fid in req["fids"]:
            f = by_fid[fid]
            cells = []
            vals = []
            for j in out:
                o = out[j].get(fid)
                if o:
                    vals.append((o["outcome"], o.get("source_outcome")))
                    flag = " · translit" if o.get("transliterated") else ""
                    flag += " · displaced" if o.get("displaced") else ""
                    so = o.get("source_outcome", "—")
                    cells.append(f"<td>sense: <b style='color:{colour.get(o['outcome'], '#000')}'>{o['outcome']}</b>"
                                 f"<br>source: <b style='color:{colour.get(so, '#000')}'>{so}</b>{flag}"
                                 f"<br><small>«{html.escape(o['english'])}» {html.escape(o['reason'])}</small></td>")
                else:
                    cells.append("<td>—</td>")
            if len(vals) == 2:
                total += 1
                agree += vals[0] == vals[1]
                agree_sense += vals[0][0] == vals[1][0]
                agree_source += vals[0][1] == vals[1][1]
            word = toks.get(f["token"], {}).get("text", "")
            rows.append(f"<tr><td class=heb>{html.escape(word)}</td><td>{f['class']}: {html.escape(f['value'])}</td>"
                        + "".join(cells) + "</tr>")
        rows.append("</table>")
        for j, a in adds.items():
            if a:
                rows.append(f"<p><b>{j} — additions:</b> " + "; ".join(
                    f"{html.escape(x['type'])}: «{html.escape(x['english'])}»" for x in a) + "</p>")
        rows.append("</section>")
    head = (f"<p>{len(ids)} pilot sentences, {total} information items. Same verdict from both judges — "
            f"sense conveyed: <b>{100 * agree_sense / max(total, 1):.0f}%</b>, "
            f"source preserved: <b>{100 * agree_source / max(total, 1):.0f}%</b>, "
            f"both: <b>{100 * agree / max(total, 1):.0f}%</b>.</p>") if total else ""
    page = ("<!doctype html><meta charset=utf-8><title>Pilot judgements</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.4}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}"
            ".src,.heb{font-size:1.2em;direction:rtl;text-align:right}.eng{background:#f3f3f3;padding:6px}"
            "section{margin-bottom:32px}</style><h1>Pilot judgements</h1>" + head + "".join(rows))
    path = root / "pilot_report.html"
    path.write_text(page, encoding="utf-8")
    print("written", path.resolve(), "|", head.replace("<b>", "").replace("</b>", "").replace("<p>", "").replace("</p>", ""))
    return path


def submit_only(judge_name: str, set_name: str) -> None:
    """Submit a set as batches (unless already submitted) and return without waiting."""
    judge = get_judge(judge_name)
    reqs = {r["id"]: r for r in _jsonl(OUT / "sets" / f"{set_name}.jsonl")}
    _batches(judge, judge_name, set_name, reqs)


def _batches(judge, judge_name: str, set_name: str, reqs: dict) -> list[str]:
    state_path = OUT / "state" / f"{judge_name}.{set_name}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(judge, reqs)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("fingerprint") != fp:
            raise SystemExit(
                f"{state_path} belongs to a different experiment (model, instructions or requests changed).\n"
                f"Its batches are not reused. Delete that file to start a new run of '{set_name}' with {judge.model}.")
        print(f"{judge_name}/{set_name}: {len(state['batches'])} batch(es) already submitted", flush=True)
        return state["batches"]
    batch_ids = judge.submit(list(reqs.values()))
    state_path.write_text(json.dumps({"fingerprint": fp, "model": judge.model, "set": set_name,
                                      "batches": batch_ids}), encoding="utf-8")
    return batch_ids


def run(judge_name: str, set_name: str) -> None:
    """Submit (unless already submitted), wait, collect; retry failed items once synchronously."""
    judge = get_judge(judge_name)
    reqs = {r["id"]: r for r in _jsonl(OUT / "sets" / f"{set_name}.jsonl")}
    batch_ids = _batches(judge, judge_name, set_name, reqs)
    print(f"{judge_name}/{set_name}: waiting for the batches (usually minutes to a few hours)…", flush=True)
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
        write_set("all", reqs)
        per: dict = {}
        for r in reqs:
            per[(r["translation"], r["book"])] = per.get((r["translation"], r["book"]), 0) + 1
        chars = sum(len(r["prompt"]) for r in reqs)
        print(len(reqs), "requests", per, f"~{chars / 3.2 / 1e6:.1f}M prompt tokens (rough)")
    elif cmd == "pilot-report":
        pilot_report()
    elif cmd == "pilot":
        pilot(argv[2], int(argv[3]) if len(argv) > 3 else 5)
    elif cmd == "submit":
        submit_only(argv[2], argv[3] if len(argv) > 3 else "main")
    elif cmd == "run":
        run(argv[2], argv[3] if len(argv) > 3 else "main")


def write_set(name: str, reqs: list[dict]) -> None:
    (OUT / "sets").mkdir(parents=True, exist_ok=True)
    with (OUT / "sets" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
