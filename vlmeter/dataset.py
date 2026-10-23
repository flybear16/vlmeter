"""Dataset loading and validation.

A dataset is a JSONL file (one item per line) or a CSV with the same columns:

  id          unique id (auto-generated from row index if missing)
  image       path to the image, relative to the dataset file
  task        "classification" | "vqa" | "checklist" (default: vqa)
  question    prompt shown to the model (not needed for checklist)
  choices     classification: option list, e.g. ["猫", "狗", "鸟"]
  answer      ground truth: option text or its letter (classification), or the
              reference answer text (vqa)
  category    optional, used for per-category breakdown in the report
  match       optional vqa scoring strategy: exact | contains | numeric | judge
  tolerance   optional numeric tolerance for match=numeric
  checks      checklist only: [{"name": "有自行车", "answer": true,
              "criteria": "可见完整的自行车或其主要部分，含部分遮挡"}] —
              one image, multiple yes/no scene checks, answered in ONE call;
              name 短名用于报告，criteria 判定标准只进提示词（可选）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_TASKS = {"classification", "vqa", "checklist"}
VALID_MATCH = {"exact", "contains", "numeric", "judge"}


@dataclass
class Check:
    name: str      # 短名：用于报告列名与结果 ID
    answer: bool
    criteria: str = ""  # 判定标准：写进提示词，对齐模型与标注员的主观标准


@dataclass
class Item:
    id: str
    image: str
    task: str = "vqa"
    question: str = ""
    choices: list[str] = field(default_factory=list)
    answer: str = ""
    category: str = "未分类"
    match: str | None = None
    tolerance: float = 0.0
    checks: list[Check] = field(default_factory=list)


def check_record_id(item_id: str, check_name: str) -> str:
    return f"{item_id}::{check_name}"


def item_record_ids(item: Item) -> list[str]:
    """Result-record ids an item expands to (checklist items yield one per check)."""
    if item.task == "checklist":
        return [check_record_id(item.id, c.name) for c in item.checks]
    return [item.id]


@dataclass
class Dataset:
    items: list[Item]
    path: Path
    fingerprint: str  # md5 of the raw file, used to key result caching

    def __len__(self) -> int:
        return len(self.items)

    @property
    def total_records(self) -> int:
        return sum(len(item_record_ids(it)) for it in self.items)


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")
    fingerprint = hashlib.md5(p.read_bytes()).hexdigest()

    rows: list[dict] = []
    if p.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            rows.append(dict(row))
    else:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p.name} line {i}: invalid JSON ({e})") from e

    items: list[Item] = []
    errors: list[str] = []
    for i, row in enumerate(rows):
        item, err = _build_item(row, i, p.parent)
        if err:
            errors.append(err)
        else:
            items.append(item)
    if errors:
        raise ValueError("dataset validation failed:\n  " + "\n  ".join(errors))
    if not items:
        raise ValueError(f"dataset {p.name} has no items")
    return Dataset(items=items, path=p, fingerprint=fingerprint)


def _build_item(row: dict, index: int, base_dir: Path) -> tuple[Item | None, str | None]:
    where = f"row {index}" + (f" (id={row.get('id')})" if row.get("id") else "")

    image = str(row.get("image") or "").strip()
    if not image:
        return None, f"{where}: missing 'image'"
    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()
    if not image_path.exists():
        return None, f"{where}: image not found: {image_path}"

    task = str(row.get("task") or "vqa").strip().lower()
    if task not in VALID_TASKS:
        return None, f"{where}: unknown task '{task}' (expected {sorted(VALID_TASKS)})"

    question = str(row.get("question") or "").strip()
    if not question and task != "checklist":
        return None, f"{where}: missing 'question'"

    checks: list[Check] = []
    if task == "checklist":
        raw_checks = row.get("checks")
        if not isinstance(raw_checks, list) or not raw_checks:
            return None, f"{where}: checklist needs a non-empty 'checks' list"
        for j, c in enumerate(raw_checks):
            if not isinstance(c, dict) or not str(c.get("name") or "").strip():
                return None, f"{where}: checks[{j}] needs a 'name'"
            ans = c.get("answer")
            if not isinstance(ans, bool):
                return None, f"{where}: checks[{j}].answer must be true/false"
            checks.append(
                Check(
                    name=str(c["name"]).strip(),
                    answer=ans,
                    criteria=str(c.get("criteria") or "").strip(),
                )
            )
    else:
        answer = str(row.get("answer") if row.get("answer") is not None else "").strip()
        if not answer:
            return None, f"{where}: missing 'answer'"

    choices_raw = row.get("choices")
    choices = [str(c).strip() for c in choices_raw if str(c).strip()] if choices_raw else []
    if task == "classification":
        if len(choices) < 2:
            return None, f"{where}: classification needs >=2 choices"
        gold = answer if answer in choices else (
            choices[ord(answer.upper()[0]) - 65] if len(answer) == 1 and "A" <= answer.upper()[0] <= "Z"
            and ord(answer.upper()[0]) - 65 < len(choices) else None
        )
        if gold is None:
            return None, f"{where}: answer '{answer}' not in choices {choices}"

    match = row.get("match")
    if match is not None:
        match = str(match).strip().lower()
        if match not in VALID_MATCH:
            return None, f"{where}: unknown match '{match}' (expected {sorted(VALID_MATCH)})"

    item = Item(
        id=str(row.get("id") or f"q{index:04d}"),
        image=str(image_path),
        task=task,
        question=question,
        choices=choices,
        answer=answer if task != "checklist" else "",
        category=str(row.get("category") or "未分类"),
        match=match,
        tolerance=float(row.get("tolerance") or 0.0),
        checks=checks,
    )
    return item, None
