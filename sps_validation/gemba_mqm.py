"""GEMBA-MQM prompt and answer scoring, reproduced from the authors' implementation.

Source: https://github.com/MicrosoftTranslator/GEMBA, commit a7a7eff8e46998447c6cbf09d06affc8f1b99ab4
(gemba/gemba_mqm_utils.py: mqm_fewshot, few_shots, parse_mqm_answer). Kocmi, T. & Federmann, C.
(2023). GEMBA-MQM: Detecting translation quality error spans with GPT-4. WMT 2023.

The prompt text, the three few-shot examples and the scoring rule (critical 25, major 5, minor 1;
at most five errors counted; score capped at 25 and negated, so 0 = no error) are copied verbatim.
This file is therefore licensed under CC BY-SA 4.0, like the GEMBA code and data it reproduces.
Only the plain-text answer path of parse_mqm_answer is needed here (the prompt asks for no JSON);
`errors()` additionally returns each error's severity, category and quoted span, which the
original also extracts (list_mqm_errors=True) but which the span analysis needs as structured data.
"""

from __future__ import annotations

import re

SYSTEM = ("You are an annotator for the quality of machine translation. Your task is to identify errors "
          "and assess the quality of the translation.")

TEMPLATE = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."""

FEW_SHOTS = [
    {
        "source_lang": "English",
        "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
        "target_lang": "German",
        "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
        "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
    },
    {
        "source_lang": "English",
        "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
        "target_lang": "Czech",
        "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
        "answer": """Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
    },
    {
        "source_lang": "Chinese",
        "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
        "target_lang": "English",
        "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
        "answer": """Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
    },
]


def messages(source_lang: str, source_seg: str, target_lang: str, target_seg: str) -> list[dict]:
    """System turn, three few-shot exchanges, then the segment to annotate (as in mqm_fewshot)."""
    out = [{"role": "system", "content": SYSTEM}]
    for shot in FEW_SHOTS:
        out.append({"role": "user", "content": TEMPLATE.format(**shot)})
        out.append({"role": "assistant", "content": shot["answer"]})
    out.append({"role": "user", "content": TEMPLATE.format(source_lang=source_lang, source_seg=source_seg,
                                                            target_lang=target_lang, target_seg=target_seg)})
    return out


def errors(answer: str) -> list[dict]:
    """Errors in a plain-text GEMBA-MQM answer, in order: severity, category line, quoted span.
    Follows parse_mqm_answer: lines are read under the last 'critical:/major:/minor:' header,
    'no-error' and empty lines are skipped, and non-translation errors count as critical."""
    out, level = [], None
    for line in answer.lower().split("\n"):
        line = line.strip()
        if "no-error" in line or "no error" in line or line == "":
            continue
        if line in ("critical:", "major:", "minor:"):
            level = line[:-1]
            continue
        if level is None:
            continue
        q = re.search(r'["“”]([^"“”]*)["“”]', line)
        out.append({"severity": "critical" if "non-translation" in line else level,
                    "category": line.split(" - ")[0].strip(), "span": q.group(1) if q else "", "line": line})
    return out


def score(answer: str | None) -> float | None:
    """GEMBA-MQM score: −(25·critical + 5·major + 1·minor) over the first five errors, capped at −25
    (order critical → major → minor, as in parse_mqm_answer). 0 means no error found."""
    if answer is None:
        return None
    total, counted = 0, 0
    found = errors(answer)
    for level in ("critical", "major", "minor"):
        for _ in (e for e in found if e["severity"] == level):
            if counted < 5:
                total += 25 if level == "critical" else 5 if level == "major" else 1
                counted += 1
    return -min(total, 25)
