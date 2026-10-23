"""Aggregate per-model result files into a Markdown comparison report."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .dataset import Dataset
from .models.base import ModelConfig
from .runner import _load_lines, result_path


def _pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.1f}%" if d else "-"


def _cost_stats(lines: list[dict], cfg: ModelConfig | None) -> dict:
    """Token/cost totals. Checklist records of one image share a single call —
    dedupe by base id so tokens are counted once per API call."""
    calls: dict[str, tuple[int, int]] = {}
    for r in lines:
        if r.get("error") or r.get("latency_ms") is None:
            continue
        base_id = str(r["id"]).split("::")[0]
        if base_id in calls:
            continue
        it, ot = r.get("input_tokens"), r.get("output_tokens")
        if it is None and ot is None:
            continue
        calls[base_id] = (it or 0, ot or 0)
    in_sum = sum(v[0] for v in calls.values())
    out_sum = sum(v[1] for v in calls.values())
    n = len(calls)
    total = cfg.estimate_cost(in_sum, out_sum) if cfg and n else None
    per_call = total / n if total is not None and n else None
    return {"n_calls": n, "in_sum": in_sum, "out_sum": out_sum,
            "total": total, "per_call": per_call}


def _fmt_cost(v: float | None, unit: str = "¥") -> str:
    if v is None:
        return "未配置单价"
    return f"{unit}{v:.4f}" if v < 1 else f"{unit}{v:.2f}"


def _agg(lines: list[dict]) -> dict:
    by_task = defaultdict(lambda: [0, 0])       # task -> [correct, scored]
    by_cat = defaultdict(lambda: [0, 0])        # category -> [correct, scored]
    lat, errors, parse_fail, unscored = [], 0, 0, 0
    for r in lines:
        if r.get("error"):
            errors += 1
            continue
        if r.get("correct") is None:
            unscored += 1
            continue
        bucket = by_task[r["task"]]
        bucket[0] += int(r["correct"])
        bucket[1] += 1
        c = by_cat[r.get("category") or "未分类"]
        c[0] += int(r["correct"])
        c[1] += 1
        if r.get("reason") == "parse_fail":
            parse_fail += 1
        if r.get("latency_ms") is not None:
            lat.append(r["latency_ms"])
    total_c = sum(v[0] for v in by_task.values())
    total_s = sum(v[1] for v in by_task.values())
    return {
        "by_task": dict(by_task),
        "by_cat": dict(by_cat),
        "total_correct": total_c,
        "total_scored": total_s,
        "avg_latency": int(sum(lat) / len(lat)) if lat else 0,
        "errors": errors,
        "parse_fail": parse_fail,
        "unscored": unscored,
    }


def _clip(s: str | None, n: int = 40) -> str:
    s = (s or "").replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def generate_report(
    dataset: Dataset,
    model_names: list[str],
    results_dir: Path,
    wrong_cases_max: int = 30,
    registry: dict[str, ModelConfig] | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_items, n_records = len(dataset), dataset.total_records
    size = f"{n_items} 图/题" if n_records == n_items else f"{n_items} 图/题，{n_records} 个判定点"
    out: list[str] = [
        f"# 多模态模型评测报告 — {dataset.path.name}",
        "",
        f"- 数据集：`{dataset.path}`（{size}，指纹 `{dataset.fingerprint[:8]}`）",
        f"- 生成时间：{now}",
        "",
        "## 总览",
        "",
        "| 模型 | 已评/总题数 | 准确率 | 分类 | 检查项 | VQA | 平均延迟 | 请求失败 | 解析失败 | 未判定 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    per_model_lines: dict[str, list[dict]] = {}
    for name in model_names:
        lines = _load_lines(result_path(name, dataset, results_dir))
        per_model_lines[name] = lines
        a = _agg(lines)
        cls = a["by_task"].get("classification", [0, 0])
        chk = a["by_task"].get("checklist", [0, 0])
        vqa = a["by_task"].get("vqa", [0, 0])
        out.append(
            f"| {name} | {a['total_scored']}/{dataset.total_records} "
            f"| **{_pct(a['total_correct'], a['total_scored'])}** "
            f"| {_pct(*cls)} | {_pct(*chk)} | {_pct(*vqa)} "
            f"| {a['avg_latency']}ms | {a['errors']} | {a['parse_fail']} | {a['unscored']} |"
        )

    out += [
        "",
        "## 成本与性能（选型权衡表）",
        "",
        "单价取自 `config/models.yaml`（请按各官网最新价核对），按本次评测的实际 token 用量估算；",
        "judge 裁判调用费用不含在内，另计。",
        "",
        "| 模型 | 准确率 | 调用次数 | tokens(入/出) | 本次总成本 | 单次调用成本 | 每1000次调用成本 | 平均延迟 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in model_names:
        a = _agg(per_model_lines[name])
        c = _cost_stats(per_model_lines[name], (registry or {}).get(name))
        acc = _pct(a["total_correct"], a["total_scored"])
        per_1k = f"{c['per_call'] * 1000:.2f}" if c["per_call"] is not None else "-"
        out.append(
            f"| {name} | {acc} | {c['n_calls']} | {c['in_sum']}/{c['out_sum']} "
            f"| {_fmt_cost(c['total'])} | {_fmt_cost(c['per_call'])} "
            f"| {per_1k} | {a['avg_latency']}ms |"
        )

    cats = {
        it.category for it in dataset.items if it.task != "checklist"
    }
    for it in dataset.items:
        if it.task == "checklist":
            cats.update(c.name for c in it.checks)  # 检查项作为分类目
    categories = sorted(cats)
    out += ["", "## 分类目准确率", "", "| 模型 | " + " | ".join(categories) + " |"]
    out.append("|---" * (len(categories) + 1) + "|")
    for name in model_names:
        a = _agg(per_model_lines[name])
        cells = [_pct(*a["by_cat"].get(c, [0, 0])) for c in categories]
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    out += ["", "## 错误样本（答错的题）", ""]
    shown = 0
    for name in model_names:
        for r in per_model_lines[name]:
            if r.get("correct") is not False or shown >= wrong_cases_max:
                continue
            reason = f"（{r['reason']}）" if r.get("reason") else ""
            base = (
                f"- **{name}** · `{r['id']}` · {r.get('category', '')} · {r['task']}{reason}\n"
                f"  - 问题：{_clip(r.get('question'), 60)}\n"
                f"  - 标准答案：`{_clip(r.get('gold'), 30)}`　模型输出：`{_clip(r.get('prediction') if r['task'] != 'checklist' else r.get('parsed'), 40)}`"
            )
            if r["task"] == "checklist" and r.get("model_reason"):
                base += f"\n  - 模型依据：{_clip(r.get('model_reason'), 80)}"
            out.append(base)
            shown += 1
    if shown == 0:
        out.append("（无答错样本）")

    out += [
        "",
        "## 失败与未判定说明",
        "",
        "- **请求失败**：API 调用重试后仍失败（网络/限流/鉴权），可用 `run` 续跑补齐。",
        "- **解析失败**：模型未按格式回答选项字母，按答错计。",
        "- **未判定**：judge 裁判调用失败，`run` 重跑会重新判定。",
        "",
    ]
    return "\n".join(out)


def report_path(dataset: Dataset, results_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"report_{dataset.path.stem}_{ts}.md"


def console_summary(
    dataset: Dataset,
    model_names: list[str],
    results_dir: Path,
    registry: dict[str, ModelConfig] | None = None,
) -> None:
    print(f"\n{'模型':<24}{'准确率':>10}{'已评':>9}{'失败':>6}{'本次成本':>14}{'每千次':>10}")
    print("-" * 78)
    for name in model_names:
        lines = _load_lines(result_path(name, dataset, results_dir))
        a = _agg(lines)
        c = _cost_stats(lines, (registry or {}).get(name))
        per_1k = f"¥{c['per_call'] * 1000:.1f}" if c["per_call"] is not None else "-"
        print(
            f"{name:<24}{_pct(a['total_correct'], a['total_scored']):>10}"
            f"{a['total_scored']:>6}/{dataset.total_records}{a['errors']:>6}"
            f"{_fmt_cost(c['total']):>14}{per_1k:>10}"
        )
