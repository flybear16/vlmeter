"""Answer parsing and scoring.

classification: parse the chosen option letter/text, exact match against gold.
vqa: strategy chosen per item ("match" field) — exact / contains / numeric,
or "judge" which defers to the LLM judge (see judge.py).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .dataset import Item
from .prompting import LETTERS

_PUNCT = re.compile(r"[\s!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~。，、；：？！“”‘’（）《》〈〉【】…—·～]+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_CN_NUM_RUN = re.compile(r"[零一二两三四五六七八九十百]+")
_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")
# "答案是B" / "故选C" / "answer is B" / "ANSWER: C" — models often enumerate
# options before concluding, so an explicit answer phrase beats raw letters.
_ANSWER_PHRASE = re.compile(
    r"(?:答案|回答|正确选项|应选|故选|选|选它|为|是|answer)\s*(?:字母|选项)?\s*[是为:：]?\s*[*\(（]?\s*([A-H])(?![A-Za-z])",
    re.IGNORECASE,
)
_BOLD_LETTER = re.compile(r"\*\*\s*([A-H])\s*\*\*")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _PUNCT.sub("", text)


def _cn_num_to_int(s: str) -> int | None:
    # Handles 零..九, 十, and compositions like 二十三 / 一百零五 (up to 百).
    if not s:
        return None
    total, num, unit = 0, 0, 1
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch == "十":
            unit = 10
            total += (num or 1) * unit
            num, unit = 0, 1
        elif ch == "百":
            unit = 100
            total += (num or 1) * unit
            num, unit = 0, 1
        else:
            return None
    return total + num * unit


def extract_numbers(text: str) -> list[float]:
    """Arabic digits take precedence; Chinese numerals are only parsed when
    the text contains no digits (avoids phrasing like 一共有5个 yielding 1)."""
    digits = [float(m) for m in _NUM.findall(unicodedata.normalize("NFKC", text))]
    if digits:
        return digits
    out: list[float] = []
    for run in _CN_NUM_RUN.findall(text):
        v = _cn_num_to_int(run)
        if v is not None:
            out.append(float(v))
    return out


@dataclass
class ScoreOutcome:
    correct: bool | None  # None = unscored (judge pending/failed)
    parsed: str           # what we understood the prediction to be
    match: str            # strategy actually used
    reason: str = ""


def parse_checklist(prediction: str, n_checks: int) -> list[tuple[bool | None, str]]:
    """Parse the per-check verdict JSON into [(verdict, reason), ...].

    Accepted value shapes (checked in order):
      {"1": {"verdict": "是", "reason": "依据"}}   ← 当前提示词要求的格式
      {"1": ["是", "依据"]}                        ← 常见变体
      {"1": "是"} / {"1": true}                    ← 无理由的简版
    Unanswered checks get (None, ""); missing JSON → all (None, "").
    """
    import json

    def _none() -> list[tuple[bool | None, str]]:
        return [(None, "")] * n_checks

    m = re.search(r"\{.*\}", prediction, re.S)
    if not m:
        return _none()
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _none()
    if not isinstance(data, dict):
        return _none()

    def _to_bool(v) -> bool | None:
        if isinstance(v, bool):
            return v
        s = normalize(str(v))
        if s in ("是", "yes", "true", "y", "correct", "符合"):
            return True
        if s in ("否", "no", "false", "n", "incorrect", "不符合"):
            return False
        return None

    out: list[tuple[bool | None, str]] = []
    for i in range(1, n_checks + 1):
        v = data.get(str(i), data.get(i))
        if isinstance(v, dict):
            verdict = _to_bool(v.get("verdict", v.get("answer", v.get("result"))))
            reason = str(v.get("reason", v.get("why", "")) or "")
        elif isinstance(v, (list, tuple)) and v:
            verdict = _to_bool(v[0]) if len(v) > 0 else None
            reason = str(v[1]) if len(v) > 1 and v[1] is not None else ""
        else:
            verdict, reason = _to_bool(v), ""
        out.append((verdict, reason))
    return out


def gold_choice_index(item: Item) -> int:
    if item.answer in item.choices:
        return item.choices.index(item.answer)
    return ord(item.answer.upper()[0]) - 65


def parse_choice(prediction: str, choices: list[str]) -> int | None:
    pred = prediction.strip()
    if not pred:
        return None

    def _idx(letter: str) -> int | None:
        i = ord(letter.upper()[0]) - 65
        return i if i < len(choices) else None

    # 1) explicit answer phrase ("答案是B", "故选C", "answer is B") — last wins
    for m in reversed(list(_ANSWER_PHRASE.finditer(pred))):
        if (i := _idx(m.group(1))) is not None:
            return i
    # 2) bolded final answer ("**B**")
    for m in reversed(list(_BOLD_LETTER.finditer(pred))):
        if (i := _idx(m.group(1))) is not None:
            return i
    # 3) last standalone letter — enumerations ("A 为圆形，B 为方形") put the
    #    conclusion last; a bare "B" reply also lands here
    letters = list(_STANDALONE_LETTER.finditer(pred))
    if letters:
        if (i := _idx(letters[-1].group(1))) is not None:
            return i
    # 4) option text contained in the prediction (longest option first)
    norm_pred = normalize(pred)
    for choice in sorted(choices, key=len, reverse=True):
        if normalize(choice) and normalize(choice) in norm_pred:
            return choices.index(choice)
    return None


def score(item: Item, prediction: str) -> ScoreOutcome:
    if item.task == "classification":
        idx = parse_choice(prediction, item.choices)
        if idx is None:
            return ScoreOutcome(False, "?", "classification", "parse_fail")
        return ScoreOutcome(
            idx == gold_choice_index(item), LETTERS[idx], "classification"
        )

    match = item.match or "contains"
    gold, pred = item.answer, prediction

    if match == "exact":
        return ScoreOutcome(normalize(pred) == normalize(gold), pred.strip(), "exact")

    if match == "contains":
        n_gold = normalize(gold)
        return ScoreOutcome(
            bool(n_gold) and n_gold in normalize(pred), pred.strip(), "contains"
        )

    if match == "numeric":
        golds = extract_numbers(gold)
        preds = extract_numbers(pred)
        ok = any(
            any(abs(g - p) <= max(item.tolerance, 1e-9) for p in preds) for g in golds
        )
        return ScoreOutcome(bool(golds) and ok, pred.strip(), "numeric")

    # "judge" is resolved by the runner (needs the judge model); scoring-only
    # callers see an unscored outcome.
    return ScoreOutcome(None, pred.strip(), "judge")
