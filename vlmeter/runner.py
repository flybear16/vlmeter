"""Async evaluation runner: concurrency per model, retries, resume support.

Results are cached at results/<model>/<dataset>__<fingerprint8>.jsonl, one JSON
line per question. Re-running skips questions already answered and scored, so
interrupted runs resume and re-runs cost nothing for unchanged items.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import asyncio as tqdmasyncio

from .dataset import Dataset, Item, check_record_id, item_record_ids
from .judge import Judge
from .models import build_model, load_registry
from .models.base import BaseVLModel, ModelConfig
from .prompting import NO, YES, build_prompt
from .retry import retry_call, short_err
from .scoring import ScoreOutcome, parse_checklist, score

RESULTS_DIR = Path("results")


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def result_path(model_name: str, dataset: Dataset, results_dir: Path = RESULTS_DIR) -> Path:
    stem = dataset.path.stem
    return results_dir / slugify(model_name) / f"{stem}__{dataset.fingerprint[:8]}.jsonl"


@dataclass
class ModelRunSummary:
    model: str
    total: int
    done_before: int = 0
    answered: int = 0
    correct: int = 0
    wrong: int = 0
    unscored: int = 0
    errors: int = 0
    skipped_no_change: int = 0


@dataclass
class RunOptions:
    models: list[str]
    data: Path
    judge: str | None = None
    limit: int | None = None
    fresh: bool = False
    registry_path: Path | None = None
    attempts: int = 3
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)


def _load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _resolve_match(item: Item, judge: Judge | None) -> str:
    if item.task == "classification":
        return "classification"
    if item.task == "checklist":
        return "checklist"
    wanted = item.match or ("judge" if judge else "contains")
    if wanted == "judge" and judge is None:
        return "contains"  # degrade gracefully when no judge is configured
    return wanted


async def _score_item(
    item: Item, prediction: str, judge: Judge | None
) -> ScoreOutcome:
    match = _resolve_match(item, judge)
    if match == "judge":
        correct, reason = await judge.verify(item.question, item.answer, prediction)
        return ScoreOutcome(correct, prediction.strip(), "judge", reason)

    outcome = score(item, prediction)
    if (
        item.task == "classification"
        and outcome.reason == "parse_fail"
        and judge is not None
    ):
        # The model described its answer instead of replying with a letter —
        # let the judge map the free-form text back to a choice.
        idx = await judge.choose(item.question, item.choices, prediction)
        if idx is not None:
            from .prompting import LETTERS
            from .scoring import gold_choice_index

            return ScoreOutcome(
                idx == gold_choice_index(item),
                LETTERS[idx],
                "classification",
                "parsed_by_judge",
            )
    return outcome


def _base_record(model_name: str, item: Item, match: str) -> dict:
    return {
        "id": item.id,
        "model": model_name,
        "task": item.task,
        "category": item.category,
        "question": item.question,
        "gold": item.answer,
        "prediction": None,
        "parsed": None,
        "correct": None,
        "match": match,
        "reason": "",
        "model_reason": None,
        "latency_ms": None,
        "tokens": None,
        "input_tokens": None,
        "output_tokens": None,
        "error": None,
    }


async def _eval_one(
    model: BaseVLModel,
    item: Item,
    judge: Judge | None,
    attempts: int,
) -> list[dict]:
    """Evaluate one dataset item; checklist items emit one record per check."""
    prompt = build_prompt(item)
    match = _resolve_match(item, judge)

    def _error_record(err: str) -> list[dict]:
        recs = []
        for rid in item_record_ids(item):
            rec = _base_record(model.name, item, match)
            rec["id"] = rid
            rec["error"] = err
            recs.append(rec)
        return recs

    try:
        async def _call():
            resp = await model.chat(item.image, prompt)
            if not resp.text:
                # Reasoning models can burn the whole max_tokens budget on
                # thinking and return empty content — retry for a fresh sample.
                raise RuntimeError(f"empty response (finish={resp.finish_reason})")
            return resp

        resp = await retry_call(_call, attempts=attempts)
    except Exception as e:  # noqa: BLE001
        return _error_record(short_err(e))

    if item.task == "checklist":
        verdicts = parse_checklist(resp.text, len(item.checks))
        records = []
        for check, (got, why) in zip(item.checks, verdicts):
            rec = _base_record(model.name, item, "checklist")
            rec.update(
                id=check_record_id(item.id, check.name),
                question=check.name,
                category=check.name,  # 分科目表 = 每个检查项的准确率
                gold=YES if check.answer else NO,
                prediction=resp.text,
                parsed=YES if got is True else NO if got is False else "?",
                correct=(got == check.answer) if got is not None else False,
                reason="" if got is not None else "parse_fail",
                model_reason=why,
                latency_ms=resp.latency_ms,
                tokens=resp.tokens,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )
            records.append(rec)
        return records

    rec = _base_record(model.name, item, match)
    rec.update(
        prediction=resp.text,
        latency_ms=resp.latency_ms,
        tokens=resp.tokens,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
    outcome = await _score_item(item, resp.text, judge)
    rec["parsed"] = outcome.parsed
    rec["correct"] = outcome.correct
    rec["match"] = outcome.match
    rec["reason"] = outcome.reason
    return [rec]


async def _run_model(
    cfg: ModelConfig,
    dataset: Dataset,
    judge: Judge | None,
    opts: RunOptions,
    registry: dict,
) -> ModelRunSummary:
    model = build_model(cfg.name, registry)
    path = result_path(cfg.name, dataset, opts.results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if opts.fresh and path.exists():
        path.unlink()

    existing = _load_lines(path)
    complete = [
        r for r in existing if r.get("error") is None and r.get("correct") is not None
    ]
    done_ids = {r["id"] for r in complete}
    # Drop stale failed/unscored lines so they are neither skipped nor counted twice.
    if len(complete) != len(existing):
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in complete),
            encoding="utf-8",
        )

    items = dataset.items[: opts.limit] if opts.limit else dataset.items
    todo = [it for it in items if not set(item_record_ids(it)) <= done_ids]

    summary = ModelRunSummary(
        model=cfg.name,
        total=len(items),
        done_before=len([it for it in items if set(item_record_ids(it)) <= done_ids]),
    )

    sem = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()
    fh = path.open("a", encoding="utf-8")

    async def worker(item: Item) -> None:
        async with sem:
            recs = await _eval_one(model, item, judge, opts.attempts)
        async with write_lock:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        summary.answered += len(recs)
        for rec in recs:
            if rec["error"] is not None:
                summary.errors += 1
            elif rec["correct"] is True:
                summary.correct += 1
            elif rec["correct"] is False:
                summary.wrong += 1
            else:
                summary.unscored += 1

    start = time.perf_counter()
    if todo:
        try:
            await tqdmasyncio.gather(
                *(worker(it) for it in todo),
                desc=f"{cfg.name} ({len(todo)} 待测)",
            )
        except AttributeError:  # older tqdm without asyncio.gather
            await asyncio.gather(*(worker(it) for it in todo))
    fh.close()
    summary.skipped_no_change = summary.done_before
    summary.elapsed_s = round(time.perf_counter() - start, 1)  # type: ignore[attr-defined]
    return summary


async def run_eval(opts: RunOptions) -> list[ModelRunSummary]:
    from .dataset import load_dataset

    dataset = load_dataset(opts.data)
    registry = load_registry(opts.registry_path)

    unknown = [m for m in opts.models if m not in registry]
    if unknown:
        raise KeyError(f"unknown models: {unknown}; available: {sorted(registry)}")

    judge: Judge | None = None
    if opts.judge:
        if opts.judge not in registry:
            raise KeyError(f"unknown judge model: {opts.judge}")
        judge = Judge(build_model(opts.judge, registry))

    summaries = await asyncio.gather(
        *(
            _run_model(registry[name], dataset, judge, opts, registry)
            for name in opts.models
        )
    )
    return list(summaries)
