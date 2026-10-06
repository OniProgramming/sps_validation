"""State-of-the-art baselines on the same planted errors and the same sentences as SATE.

Baselines (all reference-free, source-aware):
  COMETKiwi  Unbabel/wmt22-cometkiwi-da   segment score (quality estimation)
  xCOMET     Unbabel/XCOMET-XL            segment score + error spans (QE mode: no reference)
  GEMBA-MQM  the authors' prompt and scoring (gemba_mqm.py), run with the same two LLMs
             as the SATE judges, so that a difference reflects the protocol, not the model

    python -m sps_validation.baselines export                    # 1. inputs (free, seconds)
    python -m sps_validation.baselines comet cometkiwi           # 2. local / Colab (free; needs HF_TOKEN)
    python -m sps_validation.baselines comet xcomet [--gpus 1]   #    3.5B model: GPU or ≥16 GB RAM
    python -m sps_validation.baselines gemba gpt                 # 3. API (shows the cost, asks yes)
    python -m sps_validation.baselines gemba claude
    python -m sps_validation.baselines report                    # 4. tables
    python -m sps_validation.baselines report --mock             #    pipeline test on mock scores

Options: --set pairs|segments|all (default pairs: the planted errors; segments = every main
sentence, for the ranking of the translations), --batch N, --gpus N, --half (GPU, half precision),
--model NAME, --yes.

Planted-error detection (fixed before any baseline was run; docs/BASELINES.md):
  scalar   detected when the score of the perturbed text is lower than that of its identical,
           unperturbed control (Δ = control − perturbed > 0). For the LLM-based methods, whose
           verdicts vary between runs, the false-alarm rate is the same comparison between two
           judgements of the unperturbed text (SATE: control vs main run; GEMBA: control vs a
           second run of the control). The deterministic COMET models have no such floor; their
           Δ is tested against zero instead (one-sided Wilcoxon).
  span     (xCOMET, GEMBA) detected when a reported error span overlaps the edited words;
           false alarm = a span on the same words in the unperturbed control.
  SATE     targeted: the planted error's own information item is judged not fully retained
           (as in the article); scalar: sentence fidelity of perturbed < control.
"""

from __future__ import annotations

import csv
import difflib
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from . import gemba_mqm
from .judge import JUDGES, OUT, TRANSLATIONS, _jsonl, load_sources
from .report import (DIMENSIONS, DIM_LABEL, SCORE, comparisons, feature_outcomes, harmonic, load_results, ratio,
                     restrict_to_sample, sentence_rows, write_mock)
from .validate import KINDS

BASE = Path("build/baselines")
LANG = {"GEN": "Biblical Hebrew", "EPH": "Koine Greek"}
COMET_MODELS = {"cometkiwi": "Unbabel/wmt22-cometkiwi-da", "xcomet": "Unbabel/XCOMET-XL",
                "xcomet-xxl": "Unbabel/XCOMET-XXL"}
CANTILLATION = re.compile("[֑-ֽ֯׀׃-׆]")  # accents, meteg, paseq, sof pasuq …
BOOK_LABEL = {"GEN": "Genesis", "EPH": "Ephesians"}
DELETIONS = ("modifier_drop", "negation_drop")  # the planted word is gone from the perturbed text


def _plain(o):
    """numpy scalars → Python numbers for JSON."""
    return o.item() if hasattr(o, "item") else str(o)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_plain) + "\n")


def _read(path: Path) -> list[dict]:
    return _jsonl(path) if path.exists() else []



# --------------------------------------------------------------------------- meaning-preserving controls

# Edits that leave the meaning unchanged, so that a score drop on them is a false alarm. Only
# unambiguous ones: no 's/'ve contractions (it's = it is / it has; I've a son), no "let us"
# (= allow us), no change that could alter a word's sense.
CONTRACT = {"do not": "don't", "does not": "doesn't", "did not": "didn't", "is not": "isn't", "are not": "aren't",
            "was not": "wasn't", "were not": "weren't", "will not": "won't", "cannot": "can't",
            "have not": "haven't", "has not": "hasn't", "had not": "hadn't", "would not": "wouldn't",
            "should not": "shouldn't", "could not": "couldn't", "I am": "I'm", "you are": "you're",
            "we are": "we're", "they are": "they're", "I will": "I'll", "you will": "you'll", "we will": "we'll",
            "they will": "they'll", "he will": "he'll", "she will": "she'll"}
EXPAND = {v: k for k, v in CONTRACT.items()}
SPELLING = {"toward": "towards", "afterward": "afterwards", "among": "amongst", "honor": "honour", "favor": "favour",
            "neighbor": "neighbour", "labor": "labour", "color": "colour", "gray": "grey", "savior": "saviour",
            "plow": "plough", "jewelry": "jewellery"}
SPELLING |= {v: k for k, v in list(SPELLING.items())}
SPELL_SUFFIX = r"(s|ed|ing|able|ably|er|ers|hood|ful)?"
NEUTRAL_PER_TRANSLATION = None  # equalised to the smallest eligible count (see export)


def _match_case(src: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if src[:1].isupper() else repl


def neutral_edit(text: str) -> tuple[str, str] | None:
    """One meaning-preserving edit (first applicable, in a fixed order): contraction ↔ full form,
    US ↔ UK spelling, curly → straight quotation marks. Returns (new text, kind) or None."""
    for full, short in CONTRACT.items():
        m = re.search(r"\b" + re.escape(full) + r"\b", text, re.I)
        if m:
            return text[:m.start()] + _match_case(m.group(0), short) + text[m.end():], "contraction"
    for short, full in EXPAND.items():
        for form in (short, short.replace("'", "’")):
            m = re.search(r"(?<![\w’'])" + re.escape(form) + r"(?![\w’'])", text, re.I)
            if m:
                return text[:m.start()] + _match_case(m.group(0), full) + text[m.end():], "contraction"
    for a, b in SPELLING.items():
        m = re.search(r"\b" + a + SPELL_SUFFIX + r"\b", text, re.I)
        if m:
            return text[:m.start()] + _match_case(m.group(0), b) + (m.group(1) or "") + text[m.end():], "spelling"
    if re.search("[“”‘’]", text):
        return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'"), "quotes"
    return None


# --------------------------------------------------------------------------- 1. export

def source_text(req: dict, units: dict) -> str:
    """The source sentence(s) of a request as plain text. Hebrew cantillation marks are removed
    (vowels kept): they are reading accents that neural MT metrics were never trained on. The same
    text is given to every baseline."""
    return " ".join(CANTILLATION.sub("", units[u]["text"]) for u in req["units"])


WORD = re.compile(r"[\w’']+")


def _expand(text: str, s: int, e: int) -> tuple[int, int]:
    """Widen [s, e) to whole words; an empty span (a pure deletion) takes the words on both sides."""
    if s == e:
        before = list(WORD.finditer(text[:s]))
        after = WORD.search(text, e)
        return (before[-1].start() if before else s), (after.end() if after else e)
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] in "’'"):
        s -= 1
    while e < len(text) and (text[e].isalnum() or text[e] in "’'"):
        e += 1
    return s, e


def _norm(text: str) -> str:
    """Lower case with straight quotes (same length, so character offsets are kept)."""
    return text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def edit_region(control: str, perturbed: str) -> dict:
    """Where the planted error is: character spans in both texts (widened to whole words), and the
    words removed from / inserted into the control."""
    sm = difflib.SequenceMatcher(None, control, perturbed, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    if not ops:
        return {"control": [0, 0], "perturbed": [0, 0], "removed": [], "inserted": []}
    i1, i2 = min(o[1] for o in ops), max(o[2] for o in ops)
    j1, j2 = min(o[3] for o in ops), max(o[4] for o in ops)
    wc, wp = control.split(), perturbed.split()
    wm = difflib.SequenceMatcher(None, wc, wp, autojunk=False)
    removed = [w for o in wm.get_opcodes() if o[0] in ("replace", "delete") for w in wc[o[1]:o[2]]]
    inserted = [w for o in wm.get_opcodes() if o[0] in ("replace", "insert") for w in wp[o[3]:o[4]]]
    clean = lambda ws: [x for w in ws if (x := re.sub(r"^[^\w']+|[^\w']+$", "", _norm(w)))]
    removed, inserted = clean(removed), clean(inserted)
    return {"control": list(_expand(control, i1, i2)), "perturbed": list(_expand(perturbed, j1, j2)),
            "removed": [w for w in removed if w not in inserted], "inserted": [w for w in inserted if w not in removed]}


def export() -> None:
    units, _ = load_sources()
    main = {r["id"]: r for r in _jsonl(OUT / "sets" / "main.jsonl")}
    pert = _jsonl(OUT / "sets" / "perturb.jsonl")
    by_id = {q["id"]: q for q in pert}
    items, pairs = {}, []

    def item(req, role):
        items[req["id"]] = {"key": req["id"], "role": role, "translation": req["translation"], "book": req["book"],
                            "units": req["units"], "src": source_text(req, units), "mt": req["english"]}

    for q in pert:
        if q["perturbation"].get("control"):
            continue
        c = by_id.get("c" + q["id"][1:])
        if not c:
            continue
        item(q, "perturbed")
        item(c, "control")
        pairs.append({"pair": q["id"], "control": c["id"], "base": q["base_id"], "translation": q["translation"],
                      "book": q["book"], "kind": q["perturbation"]["kind"], "target": q["perturbation"]["target"],
                      "change": q["perturbation"]["change"], "edit": edit_region(c["english"], q["english"])})
    for r in main.values():  # the main set is the judged sample (sampled Genesis + all of Ephesians)
        item(r, "segment")
    # meaning-preserving variants of the controls: the same number per translation, chosen with a fixed seed
    eligible = defaultdict(list)
    for p in pairs:
        c = by_id[p["control"]]
        e = neutral_edit(c["english"])
        if e:
            eligible[p["translation"]].append((c, e))
    n_each = min((len(v) for v in eligible.values()), default=0)
    neutral = []
    rng = random.Random(20261006)
    for t in sorted(eligible):
        for c, (new, kind) in sorted(rng.sample(eligible[t], n_each), key=lambda x: x[0]["id"]):
            key = "n" + c["id"]
            items[key] = {"key": key, "role": "neutral", "translation": t, "book": c["book"], "units": c["units"],
                          "src": source_text(c, units), "mt": new}
            neutral.append({"neutral": key, "control": c["id"], "translation": t, "book": c["book"], "kind": kind,
                            "edit": edit_region(c["english"], new)})
    _write_jsonl(BASE / "items.jsonl", items.values())
    _write_jsonl(BASE / "pairs.jsonl", pairs)
    _write_jsonl(BASE / "neutral.jsonl", neutral)
    n = defaultdict(int)
    for i in items.values():
        n[i["role"]] += 1
    print(f"exported {len(pairs)} planted-error pairs, {n['neutral']} meaning-preserving controls "
          f"({n_each} per translation) and {n['segment']} main segments ({len(items)} texts) to {BASE}/")


def scores_path(method: str, which: str) -> Path:
    """The meaning-preserving controls go to their own file (<method>.neutral.jsonl), so that adding
    them later never overwrites the scores already obtained; report merges both."""
    return BASE / "scores" / (f"{method}.neutral.jsonl" if which == "neutral" else f"{method}.jsonl")


def selected(which: str) -> list[dict]:
    items = _read(BASE / "items.jsonl")
    if not items:
        raise SystemExit("run `python -m sps_validation.baselines export` first")
    roles = {"pairs": ("perturbed", "control"), "segments": ("segment",), "neutral": ("neutral",),
             "all": ("perturbed", "control", "segment", "neutral")}
    return [i for i in items if i["role"] in roles[which]]


# --------------------------------------------------------------------------- 2. COMET models

def run_comet(name: str, which: str, batch: int, gpus: int | None, model_name: str | None,
              half: bool = False) -> None:
    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        raise SystemExit(f"cannot import COMET ({type(e).__name__}: {e}). "
                         "pip install unbabel-comet; see docs/BASELINES.md")
    model_id = model_name or COMET_MODELS[name]
    path = scores_path(name, which)
    done = {r["key"] for r in _read(path)}
    todo = [i for i in selected(which) if i["key"] not in done]
    if not todo:
        print(f"{name}: nothing to do ({len(done)} texts already scored)")
        return
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else 0
    model = load_from_checkpoint(download_model(model_id))
    if half and gpus:
        model = model.half()  # fallback when a 16 GB GPU runs out of memory with xCOMET-XL
    tok = model.encoder.tokenizer
    limit = model.encoder.max_positions
    print(f"{name} ({model_id}): {len(todo)} texts, gpus={gpus}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(todo), 200):  # in chunks, so an interruption keeps what was scored
        chunk = todo[k:k + 200]
        out = model.predict([{"src": i["src"], "mt": i["mt"]} for i in chunk], batch_size=batch, gpus=gpus,
                            progress_bar=True)
        meta = out.get("metadata") or {}
        spans = meta.get("error_spans") if hasattr(meta, "get") else None
        with path.open("a", encoding="utf-8") as f:
            for n, i in enumerate(chunk):
                # the model reads translation + source in one sequence of at most `limit` tokens;
                # longer inputs are cut at the end, i.e. in the source
                length = len(tok(i["mt"])["input_ids"]) + len(tok(i["src"])["input_ids"])
                rec = {"key": i["key"], "model": model_id, "score": float(out.scores[n]),
                       "tokens": length, "truncated": length > limit}
                if spans is not None:
                    rec["spans"] = [{"start": s["start"], "end": s["end"], "severity": s["severity"],
                                     "text": s["text"], "confidence": round(float(s["confidence"]), 4)}
                                    for s in spans[n]]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {min(k + 200, len(todo))}/{len(todo)}", flush=True)


# --------------------------------------------------------------------------- 3. GEMBA-MQM

GEMBA_OUTPUT = {"claude": 200, "gpt": 1500}  # expected output tokens per call (GPT: incl. reasoning), for the estimate


def gemba_client(judge: str, model: str):
    if judge == "claude":
        import anthropic

        client = anthropic.Anthropic(max_retries=8)

        def call(msgs: list[dict], attempt: int) -> tuple[str | None, dict]:
            r = client.messages.create(model=model, max_tokens=1024, temperature=0 if attempt == 0 else 1,
                                       system=msgs[0]["content"], messages=msgs[1:])
            text = "".join(b.text for b in r.content if b.type == "text") or None
            return text, {"input": r.usage.input_tokens, "output": r.usage.output_tokens, "served_model": r.model}
    else:
        import openai

        client = openai.OpenAI(max_retries=8)
        effort = os.environ.get("JUDGE_GPT_EFFORT", "medium")

        def call(msgs: list[dict], attempt: int) -> tuple[str | None, dict]:
            body = {"model": model, "input": msgs, "max_output_tokens": 16000}
            if model.startswith(("gpt-5", "o")):
                body["reasoning"] = {"effort": effort}  # same setting as the SATE judge; no temperature
            else:
                body["temperature"] = 0 if attempt == 0 else 1
            r = client.responses.create(**body)
            u = r.usage
            return (r.output_text or None), {"input": u.input_tokens if u else None,
                                             "output": u.output_tokens if u else None, "served_model": r.model}
    return call


def gemba_jobs(which: str) -> list[dict]:
    """One job per text; for the planted errors each control is judged twice (key suffix '#2'),
    which measures how often GEMBA lowers its verdict on an unchanged text (false alarms)."""
    jobs = []
    for i in selected(which):
        jobs.append(i)
        if i["role"] == "control":
            jobs.append(dict(i, key=i["key"] + "#2"))
    return jobs


def run_gemba(judge: str, which: str, model: str | None, yes: bool) -> None:
    model = model or JUDGES[judge]
    path = scores_path(f"gemba-{judge}", which)
    done = {r["key"] for r in _read(path) if r.get("status") == "ok"}
    todo = [j for j in gemba_jobs(which) if j["key"] not in done]
    if not todo:
        print(f"gemba-{judge}: nothing to do ({len(done)} texts already scored)")
        return
    prices = json.loads(Path("config/prices.json").read_text(encoding="utf-8"))
    p = prices.get(model)
    chars = sum(len(json.dumps(gemba_mqm.messages(LANG[j["book"]], j["src"], "English", j["mt"]), ensure_ascii=False))
                for j in todo)
    est_in, est_out = chars / 2.5, GEMBA_OUTPUT[judge] * len(todo)  # Hebrew/Greek: ~2.5 characters per token
    cost = (est_in * p["input"] + est_out * p["output"]) / 1e6 if p else None
    print(f"GEMBA-MQM with {model}: {len(todo)} calls"
          + (f", estimated cost ≈ ${cost:.2f} (standard API price, not batch)" if cost is not None else ""))
    if not yes and input("Type yes to start: ").strip().lower() != "yes":
        print("Stopped. Nothing was spent.")
        return
    call = gemba_client(judge, model)
    lock = threading.Lock()
    path.parent.mkdir(parents=True, exist_ok=True)

    def work(job):
        msgs = gemba_mqm.messages(LANG[job["book"]], job["src"], "English", job["mt"])
        rec = {"key": job["key"], "model": model}
        try:
            for attempt in range(2):  # GEMBA retries a request that returned no answer
                answer, usage = call(msgs, attempt)
                if answer:
                    break
            rec |= {"status": "ok" if answer else "empty", "answer": answer, "usage": usage,
                    "errors": gemba_mqm.errors(answer) if answer else [], "score": gemba_mqm.score(answer)}
        except Exception as e:  # recorded and retried on the next run
            rec |= {"status": "failed", "error": str(e)[:500]}
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["status"]

    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for status in pool.map(work, todo):
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    bad = sum(1 for r in _latest(path).values() if r.get("status") != "ok")
    print(f"gemba-{judge}: done; {bad} texts without an answer (run the same command again to retry them)")


def _latest(path: Path) -> dict[str, dict]:
    """Last record per key (a failed call retried later is superseded)."""
    out = {}
    for r in _read(path):
        out[r["key"]] = r
    return out


# --------------------------------------------------------------------------- 4. report

def overlap(a: int, b: int, s: int, e: int) -> bool:
    return a < e and s < b


def xcomet_span_hit(rec: dict | None, region: list[int]) -> bool | None:
    if not rec or "spans" not in rec:
        return None
    return any(overlap(s["start"], s["end"], *region) for s in rec["spans"])


def gemba_span_hit(rec: dict | None, text: str, region: list[int], removed: list[str] | None = None) -> bool | None:
    """An error whose quoted span lies on the edited words; for a deletion (modifier, negation)
    also an error that quotes the deleted word, which no longer occurs in the text."""
    if not rec or rec.get("status") != "ok":
        return None
    low = _norm(text)
    for e in rec["errors"]:
        q = _norm(e["span"].strip())
        if not q:
            continue
        if any(overlap(m.start(), m.end(), *region) for m in re.finditer(re.escape(q), low)):
            return True
        if removed and set(WORD.findall(q)) & set(removed):
            return True
    return False


def request_fidelity(recs: list[dict], key: str) -> float | None:
    """SATE fidelity of one request (all its features), with the verdicts of `recs` (one per
    judge) combined as in the main scoring."""
    recs = [r for r in recs if r and r["result"].get("status") == "ok"]
    if not recs:
        return None
    outs = [feature_outcomes(r) for r in recs]
    S = D = N = 0.0
    for fid in set().union(*outs):
        ans = [o[fid][key] for o in outs if fid in o and o[fid].get(key) in SCORE]
        if not ans:
            continue
        S += sum(SCORE[a] for a in ans) / len(ans)
        D += sum(a == "distorted" for a in ans) / len(ans)
        N += 1
    if not N:
        return None
    U = float(np.mean([sum(a["type"] == "unsupported" for a in r["result"].get("additions", [])) for r in recs]))
    return float(harmonic(ratio(S, S + D + U), S / N))


def sate_targeted(pair: dict, pert: dict, main: dict, judges: tuple, key: str) -> tuple[bool, bool] | None:
    """(detected, false alarm) by the article's definition, with judges combined by their mean."""
    unsupported = lambda r: sum(a["type"] == "unsupported" for a in r["result"].get("additions", []))
    recs = []
    for j in judges:
        r = (pert.get(j, {}).get(pair["pair"]), pert.get(j, {}).get(pair["control"]), main.get(j, {}).get(pair["base"]))
        if not all(r) or any(x["result"].get("status") != "ok" for x in r):
            return None
        recs.append(r)
    if pair["kind"] == "addition":
        mean = lambda i: sum(unsupported(r[i]) for r in recs) / len(recs)
        return mean(0) > mean(2), mean(1) > mean(2)
    outs = [tuple(feature_outcomes(x).get(pair["target"], {}).get(key) for x in r) for r in recs]
    if any(o not in SCORE for oo in outs for o in oo) or any(o[2] != "retained" for o in outs):
        return None
    score = lambda i: sum(SCORE[o[i]] for o in outs) / len(outs)
    return score(0) < 1, score(1) < 1


def evaluate(pairs: list[dict], items: dict, scores: dict, sate: dict) -> dict:
    """Per method: one record per pair with detected / false alarm / Δ (None = not assessable)."""
    out = defaultdict(list)
    pert, main = sate["pert"], sate["main"]
    judges = tuple(j for j in ("claude", "gpt") if j in pert)
    combos = [(j,) for j in judges] + ([judges] if len(judges) > 1 else [])
    name = lambda jj: "SATE " + ("Claude+GPT" if len(jj) > 1 else {"claude": "Claude", "gpt": "GPT"}[jj[0]])
    for p in pairs:
        base = {"pair": p["pair"], "kind": p["kind"], "translation": p["translation"], "book": p["book"]}
        for jj in combos:
            for d, key in DIMENSIONS.items():
                t = sate_targeted(p, pert, main, jj, key)
                out[f"{name(jj)} · targeted · {d}"].append(base | (
                    {"detected": t[0], "alarm": t[1]} if t else {"detected": None, "alarm": None}))
            fp = request_fidelity([pert[j].get(p["pair"]) for j in jj], "outcome")
            fc = request_fidelity([pert[j].get(p["control"]) for j in jj], "outcome")
            fo = request_fidelity([main[j].get(p["base"]) for j in jj], "outcome")
            ok = None not in (fp, fc)
            out[f"{name(jj)} · sentence fidelity · sense"].append(base | {
                "delta": fc - fp if ok else None, "detected": fc > fp if ok else None,
                "alarm": (fo > fc) if ok and fo is not None else None,
                "null_delta": fo - fc if ok and fo is not None else None})
        for metric in ("cometkiwi", "xcomet", "xcomet-xxl"):
            s = scores.get(metric)
            if not s:
                continue
            a, b = s.get(p["pair"]), s.get(p["control"])
            ok = a is not None and b is not None
            label = {"cometkiwi": "COMETKiwi", "xcomet": "xCOMET-XL", "xcomet-xxl": "xCOMET-XXL"}[metric]
            out[f"{label} · score"].append(base | {
                "delta": b["score"] - a["score"] if ok else None, "detected": b["score"] > a["score"] if ok else None,
                "alarm": None, "truncated": (a["truncated"] or b["truncated"]) if ok else None})
            if metric.startswith("xcomet"):
                hp = xcomet_span_hit(a, p["edit"]["perturbed"])
                hc = xcomet_span_hit(b, p["edit"]["control"])
                out[f"{label} · error span"].append(base | {"detected": hp, "alarm": hc})
        for j in ("claude", "gpt"):
            s = scores.get(f"gemba-{j}")
            if not s:
                continue
            a, b, b2 = s.get(p["pair"]), s.get(p["control"]), s.get(p["control"] + "#2")
            ok = all(x and x.get("status") == "ok" for x in (a, b))
            label = f"GEMBA-MQM {'Claude' if j == 'claude' else 'GPT'}"
            alarm = (b2["score"] < b["score"]) if ok and b2 and b2.get("status") == "ok" else None
            out[f"{label} · score"].append(base | {
                "delta": b["score"] - a["score"] if ok else None, "detected": b["score"] > a["score"] if ok else None,
                "alarm": alarm, "null_delta": b["score"] - b2["score"] if alarm is not None else None})
            deleted = p["edit"]["removed"] if p["kind"] in DELETIONS else None
            hp = gemba_span_hit(a, items[p["pair"]]["mt"], p["edit"]["perturbed"], deleted)
            hc = gemba_span_hit(b, items[p["control"]]["mt"], p["edit"]["control"])
            out[f"{label} · error span"].append(base | {"detected": hp, "alarm": hc})
    return out


def neutral_deltas(neutral: list[dict], scores: dict) -> dict[str, list[dict]]:
    """Score change on meaning-preserving edits, per method: Δ = control − edited (> 0 = a drop)."""
    out = {}
    labels = {"cometkiwi": "COMETKiwi · score", "xcomet": "xCOMET-XL · score", "xcomet-xxl": "xCOMET-XXL · score",
              "gemba-claude": "GEMBA-MQM Claude · score", "gemba-gpt": "GEMBA-MQM GPT · score"}
    for m, s in scores.items():
        rows = []
        for n in neutral:
            a, b = s.get(n["control"]), s.get(n["neutral"])
            if not a or not b or a.get("status", "ok") != "ok" or b.get("status", "ok") != "ok":
                continue
            rows.append({"kind": n["kind"], "translation": n["translation"], "delta": a["score"] - b["score"],
                         "n": n, "rec": b})
        if rows:
            out[labels.get(m, m)] = rows
    return out


def calibration(records: dict, neutral_rows: dict, alpha: float = 0.05) -> list[dict]:
    """Detection at matched specificity. For each score-based method, a score drop Δ counts as a
    detection only above the threshold τ that at most `alpha` of the null comparisons exceed. Null
    comparisons are same-meaning texts: meaning-preserving edits (all methods with such scores) and,
    for methods whose verdicts vary between runs, two judgements of the identical text."""
    out = []
    for method, rs in records.items():
        if not any(r.get("delta") is not None for r in rs):
            continue
        null = [r["null_delta"] for r in rs if r.get("null_delta") is not None]
        null += [x["delta"] for x in neutral_rows.get(method, [])]
        if len(null) < 20:
            continue
        null = np.array(null, dtype=float)
        cands = sorted({0.0} | {float(v) for v in null if v > 0})
        tau = next(t for t in cands if np.mean(null > t) <= alpha) if any(np.mean(null > t) <= alpha for t in cands) \
            else max(cands)
        row = {"method": method, "tau": round(tau, 4), "null_n": len(null),
               "null_false_alarm": round(float(np.mean(null > tau)), 3),
               "null_any_drop": round(float(np.mean(null > 0)), 3),
               "neutral_n": len(neutral_rows.get(method, [])),
               "neutral_any_drop": round(float(np.mean([x["delta"] > 0 for x in neutral_rows[method]])), 3)
               if neutral_rows.get(method) else None}
        for k in KINDS + ["all"]:
            d = np.array([r["delta"] for r in rs if r.get("delta") is not None and (k == "all" or r["kind"] == k)])
            row[k] = round(float(np.mean(d > tau)), 3) if len(d) else None
        out.append(row)
    return out


def neutral_spans(neutral: list[dict], items: dict, scores: dict) -> list[dict]:
    """Error spans placed on the edited words of a meaning-preserving edit (false alarms of the
    localisation). Quote-mark edits touch the whole quotation and are left out."""
    out = []
    for m, label in (("xcomet", "xCOMET-XL · error span"), ("xcomet-xxl", "xCOMET-XXL · error span"),
                     ("gemba-gpt", "GEMBA-MQM GPT · error span"), ("gemba-claude", "GEMBA-MQM Claude · error span")):
        s = scores.get(m)
        if not s:
            continue
        hits = []
        for n in neutral:
            if n["kind"] == "quotes" or n["neutral"] not in s:
                continue
            rec = s[n["neutral"]]
            h = xcomet_span_hit(rec, n["edit"]["perturbed"]) if m.startswith("xcomet") else \
                gemba_span_hit(rec, items[n["neutral"]]["mt"], n["edit"]["perturbed"])
            if h is not None:
                hits.append(h)
        if hits:
            out.append({"method": label, "n": len(hits), "false_alarm": round(sum(hits) / len(hits), 3)})
    return out


def rate_table(records: dict) -> list[dict]:
    rows = []
    for method, rs in records.items():
        row = {"method": method}
        for k in KINDS + ["all"]:
            sub = [r for r in rs if (k == "all" or r["kind"] == k) and r["detected"] is not None]
            alarms = [r["alarm"] for r in sub if r.get("alarm") is not None]
            deltas = np.array([r["delta"] for r in sub if r.get("delta") is not None], dtype=float)
            p = None
            if len(deltas) and np.any(deltas != 0):
                p = float(stats.wilcoxon(deltas, alternative="greater", zero_method="wilcox").pvalue) \
                    if np.count_nonzero(deltas) >= 2 else None
            row[k] = {"n": len(sub), "rate": round(sum(r["detected"] for r in sub) / len(sub), 3) if sub else None,
                      "false_alarm_rate": round(sum(alarms) / len(alarms), 3) if alarms else None,
                      "mean_delta": round(float(deltas.mean()), 4) if len(deltas) else None,
                      "p_delta_gt_0": p}
        trunc = [r["truncated"] for r in rs if r.get("truncated") is not None]
        row["truncated"] = round(sum(trunc) / len(trunc), 3) if trunc else None
        rows.append(row)
    return rows


def mcnemar(records: dict, ref: str) -> list[dict]:
    """Exact McNemar test of each method against `ref` on the pairs both could assess."""
    out = []
    a = {r["pair"]: r["detected"] for r in records.get(ref, []) if r["detected"] is not None}
    for method, rs in records.items():
        if method == ref:
            continue
        b = {r["pair"]: r["detected"] for r in rs if r["detected"] is not None}
        common = a.keys() & b.keys()
        only_ref = sum(a[k] and not b[k] for k in common)
        only_m = sum(b[k] and not a[k] for k in common)
        p = stats.binomtest(min(only_ref, only_m), only_ref + only_m, 0.5).pvalue if only_ref + only_m else 1.0
        out.append({"method": method, "pairs": len(common), "ref_rate": round(sum(a[k] for k in common) / len(common), 3)
                    if common else None, "method_rate": round(sum(b[k] for k in common) / len(common), 3) if common else None,
                    "only_ref": only_ref, "only_method": only_m, "p": float(p)})
    return out


def rankings(items: dict, scores: dict, sate_rows: dict) -> list[dict]:
    """Order of the translations per book: SATE (both dimensions) and each baseline. A baseline
    scores a segment (one alignment group); its score is given to each source sentence of the group,
    so that all methods are compared on the same sentences (same tests as Table 4)."""
    out = []
    for d, rows in sate_rows.items():
        out.append({"method": f"SATE · {DIM_LABEL[d]}", "rows": rows})
    labels = {"cometkiwi": "COMETKiwi", "xcomet": "xCOMET-XL", "xcomet-xxl": "xCOMET-XXL",
              "gemba-claude": "GEMBA-MQM Claude", "gemba-gpt": "GEMBA-MQM GPT"}
    keep = {r["sentence"] for rows in sate_rows.values() for r in rows}
    for m, s in scores.items():
        rows = []
        for i in items.values():
            if i["role"] != "segment" or i["key"] not in s:
                continue
            v = s[i["key"]]
            if v.get("status", "ok") != "ok" or v.get("score") is None:
                continue
            for u in i["units"]:
                if u in keep:
                    rows.append({"book": i["book"], "sentence": u, "translation": i["translation"],
                                 "fidelity": float(v["score"])})
        if rows:
            out.append({"method": labels.get(m, m), "rows": rows})
    result = []
    ref = {}
    for o in out:
        comp = comparisons(o["rows"])
        for b in ("GEN", "EPH"):
            means = {t: float(np.mean([r["fidelity"] for r in o["rows"] if r["book"] == b and r["translation"] == t]))
                     for t in TRANSLATIONS if any(r["book"] == b and r["translation"] == t for r in o["rows"])}
            if len(means) < 2:
                continue
            c = comp.get(b)
            rec = {"method": o["method"], "book": b, "means": {t: round(v, 4) for t, v in means.items()},
                   "order": " > ".join(sorted(means, key=lambda t: -means[t])),
                   "W": c["kendall_W"] if c else None,
                   "significant": sum(p["p_holm"] < 0.05 for p in c["pairwise"]) if c else None,
                   "sentences": c["sentences"] if c else None}
            if o["method"].startswith("SATE"):
                ref[(o["method"], b)] = means
            result.append(rec)
    for rec in result:  # agreement of each order with SATE's two orders (Kendall's τ over 4 translations)
        for d in DIMENSIONS:
            r = ref.get((f"SATE · {DIM_LABEL[d]}", rec["book"]))
            if r and len(rec["means"]) == len(r):
                tau = stats.kendalltau([r[t] for t in TRANSLATIONS], [rec["means"][t] for t in TRANSLATIONS]).statistic
                rec[f"tau_{d}"] = round(float(tau), 3)
    return result


def load_scores(root: Path) -> dict:
    """{method: {key: record}}; <method>.jsonl and <method>.neutral.jsonl are merged."""
    out = defaultdict(dict)
    for path in sorted(root.glob("*.jsonl")):
        out[path.name.split(".")[0]].update(_latest(path))
    return dict(out)


def report(mock: bool) -> None:
    pairs = _read(BASE / "pairs.jsonl")
    items = {i["key"]: i for i in _read(BASE / "items.jsonl")}
    if not pairs:
        raise SystemExit("run `python -m sps_validation.baselines export` first")
    if mock:
        root, sroot, out, label = OUT / "results_mock", BASE / "scores_mock", Path("build/report/mock/baselines"), \
            "MOCK DATA — pipeline test, not a result"
        if not (root / "claude" / "main.jsonl").exists():
            write_mock(root)
        write_mock_scores(items, sroot)
    else:
        root, sroot, out, label = OUT / "results", BASE / "scores", Path("build/report/baselines"), ""
    sate = {"pert": load_results(root, "perturb"), "main": load_results(root, "main")}
    scores = load_scores(sroot)
    records = evaluate(pairs, items, scores, sate)
    table = rate_table(records)
    neutral = _read(BASE / "neutral.jsonl")
    calib = calibration(records, neutral_deltas(neutral, scores))
    nspans = neutral_spans(neutral, items, scores)
    ref = "SATE Claude+GPT · targeted · sense"  # SATE as published: both judges combined
    tests = mcnemar(records, ref) if ref in records else []
    units, feats = load_sources()
    main_reqs = _jsonl(OUT / "sets" / "main.jsonl")
    sate_rows = {d: restrict_to_sample(sentence_rows(main_reqs, sate["main"], units, feats, k))
                 for d, k in DIMENSIONS.items()} if sate["main"] else {}
    ranks = rankings(items, scores, sate_rows)
    res = {"label": label, "methods": list(records), "detection": table, "mcnemar_vs": ref, "mcnemar": tests,
           "calibrated": calib, "neutral_spans": nspans, "neutral_controls": len(neutral),
           "rankings": ranks, "pairs": len(pairs)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=_plain), encoding="utf-8")
    _write_jsonl(out / "pair_records.jsonl", ({"method": m} | r for m, rs in records.items() for r in rs))
    with (out / "baselines_detection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [f"{k} detection" for k in KINDS] + ["All: detection", "All: false alarms",
                                                                     "All: mean Δ", "All: p(Δ>0)", "Pairs", "Truncated"])
        for r in table:
            a = r["all"]
            w.writerow([r["method"]] + [r[k]["rate"] for k in KINDS] +
                       [a["rate"], a["false_alarm_rate"], a["mean_delta"], a["p_delta_gt_0"], a["n"], r["truncated"]])
    with (out / "baselines_calibrated.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [f"{k} detection" for k in KINDS] + ["All: detection", "Threshold τ",
                                                                     "False alarms at τ", "Null comparisons",
                                                                     "Neutral edits with any drop"])
        for r in calib:
            w.writerow([r["method"]] + [r[k] for k in KINDS] + [r["all"], r["tau"], r["null_false_alarm"], r["null_n"],
                                                                r["neutral_any_drop"]])
    with (out / "baselines_rankings.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Book"] + list(TRANSLATIONS) + ["Order", "Kendall W", "Significant pairs",
                                                              "τ with SATE sense", "τ with SATE source"])
        for r in ranks:
            w.writerow([r["method"], BOOK_LABEL[r["book"]]] + [r["means"].get(t, "") for t in TRANSLATIONS] +
                       [r["order"], r["W"], r["significant"], r.get("tau_sense", ""), r.get("tau_source", "")])
    (out / "baselines.md").write_text(markdown(res), encoding="utf-8")
    print(f"{len(table)} methods compared on {len(pairs)} planted-error pairs; results in {out}/")


def markdown(res: dict) -> str:
    f = lambda x: "—" if x is None else f"{x:.2f}"
    L = [f"# SATE and state-of-the-art baselines{' — ' + res['label'] if res['label'] else ''}", "",
         f"Planted-error benchmark: {res['pairs']} perturbed/control pairs (6 error types × 4 translations). "
         "Detection rate per error type; false alarms on the unperturbed control where the method can have them; "
         "for score-based methods also the mean score drop Δ = control − perturbed and the one-sided Wilcoxon "
         "p-value for Δ > 0. Definitions: module docstring of baselines.py and docs/BASELINES.md.", "",
         "| Method | " + " | ".join(KINDS) + " | All | False alarms | Mean Δ | p(Δ>0) | Pairs | Truncated |",
         "|---|" + "---|" * (len(KINDS) + 6)]
    for r in res["detection"]:
        a = r["all"]
        p = "—" if a["p_delta_gt_0"] is None else format(a["p_delta_gt_0"], ".3g")
        L.append(f"| {r['method']} | " + " | ".join(f(r[k]["rate"]) for k in KINDS) +
                 f" | {f(a['rate'])} | {f(a['false_alarm_rate'])} | "
                 f"{'—' if a['mean_delta'] is None else a['mean_delta']} | {p} | {a['n']} | "
                 f"{'—' if r['truncated'] is None else r['truncated']} |")
    if res["mcnemar"]:
        L += ["", f"Paired comparison with {res['mcnemar_vs']} (exact McNemar test on the pairs both methods assess):",
              "", "| Method | Pairs | SATE rate | Method rate | Only SATE | Only method | p |", "|---|---|---|---|---|---|---|"]
        for t in res["mcnemar"]:
            L.append(f"| {t['method']} | {t['pairs']} | {f(t['ref_rate'])} | {f(t['method_rate'])} | "
                     f"{t['only_ref']} | {t['only_method']} | {t['p']:.3g} |")
    if res.get("calibrated"):
        L += ["", "## Detection at matched specificity (at most 5% false alarms)", "",
              f"Same-meaning comparisons: {res['neutral_controls']} meaning-preserving edits of the controls "
              "(contraction ↔ full form, US ↔ UK spelling, curly → straight quotes; same number per translation) "
              "and, for methods whose verdicts vary between runs, two judgements of the identical text. τ is the "
              "smallest score drop that at most 5% of these comparisons exceed; a planted error counts as detected "
              "when its drop exceeds τ. SATE's targeted verdicts need no threshold (false alarms 1.5%).", "",
              "| Method | " + " | ".join(KINDS) + " | All | τ | False alarms at τ | Null comparisons | Neutral edits: any drop |",
              "|---|" + "---|" * (len(KINDS) + 5)]
        for r in res["calibrated"]:
            L.append(f"| {r['method']} | " + " | ".join(f(r[k]) for k in KINDS) + f" | {f(r['all'])} | {r['tau']} | "
                     f"{r['null_false_alarm']} | {r['null_n']} | "
                     f"{'—' if r['neutral_any_drop'] is None else r['neutral_any_drop']} |")
    if res.get("neutral_spans"):
        L += ["", "Error spans placed on the edited words of a meaning-preserving edit (contraction and spelling "
                  "edits; localisation false alarms):", "", "| Method | Edits | False alarms |", "|---|---|---|"]
        for r in res["neutral_spans"]:
            L.append(f"| {r['method']} | {r['n']} | {f(r['false_alarm'])} |")
    if res["rankings"]:
        L += ["", "## Order of the translations", "",
              "Mean score per source sentence (SATE: fidelity; baselines: their own scale, so only the order and the "
              "tests are comparable). W and significant pairs: Friedman / Wilcoxon-Holm over sentences, as in Table 4. "
              "τ: Kendall's rank correlation of the four means with SATE's.", "",
              "| Method | Book | " + " | ".join(TRANSLATIONS) + " | Order | W | Sig. pairs | τ sense | τ source |",
              "|---|---|" + "---|" * len(TRANSLATIONS) + "---|---|---|---|---|"]
        for r in res["rankings"]:
            L.append(f"| {r['method']} | {BOOK_LABEL[r['book']]} | " +
                     " | ".join(str(r["means"].get(t, "—")) for t in TRANSLATIONS) +
                     f" | {r['order']} | {r['W']} | {r['significant']}/6 | {r.get('tau_sense', '—')} | "
                     f"{r.get('tau_source', '—')} |")
    return "\n".join(L) + "\n"


def write_mock_scores(items: dict, root: Path) -> None:
    """Random scores in the baselines' formats (pipeline test only)."""
    rng = random.Random(3)
    root.mkdir(parents=True, exist_ok=True)
    keys = list(items)
    for m in ("cometkiwi", "xcomet"):
        rows = []
        for k in keys:
            mt = items[k]["mt"]
            s = rng.randrange(max(len(mt), 1))
            rec = {"key": k, "model": "MOCK", "score": rng.random(), "tokens": 100, "truncated": rng.random() < 0.1}
            if m == "xcomet":
                rec["spans"] = [{"start": s, "end": min(s + 8, len(mt)), "severity": "minor", "text": mt[s:s + 8],
                                 "confidence": 0.5}] if rng.random() < 0.5 else []
            rows.append(rec)
        _write_jsonl(root / f"{m}.jsonl", rows)
    for j in ("claude", "gpt"):
        rows = []
        for k in keys + [k + "#2" for k in keys if items[k]["role"] == "control"]:
            words = items[k.split("#")[0]]["mt"].split()
            ans = "Critical:\nno-error\nMajor:\n" + (f'accuracy/mistranslation - "{rng.choice(words)}"\n'
                                                       if words and rng.random() < 0.4 else "no-error\n") + "Minor:\nno-error\n"
            rows.append({"key": k, "model": "MOCK", "status": "ok", "answer": ans, "errors": gemba_mqm.errors(ans),
                         "score": gemba_mqm.score(ans)})
        _write_jsonl(root / f"gemba-{j}.jsonl", rows)


# --------------------------------------------------------------------------- CLI

def _opt(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "help"
    which = _opt(argv, "--set", "pairs")
    if cmd == "export":
        export()
    elif cmd == "comet":
        name = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else "cometkiwi"
        g = _opt(argv, "--gpus")
        run_comet(name, which, int(_opt(argv, "--batch", 8)), int(g) if g is not None else None, _opt(argv, "--model"),
                  "--half" in argv)
    elif cmd == "gemba":
        judge = argv[2] if len(argv) > 2 and argv[2] in ("claude", "gpt") else "gpt"
        run_gemba(judge, which, _opt(argv, "--model"), "--yes" in argv)
    elif cmd == "report":
        report("--mock" in argv)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
