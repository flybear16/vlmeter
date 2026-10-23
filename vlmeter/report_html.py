"""Self-contained HTML report: single file, inline CSS, no external deps."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from . import __version__
from .dataset import Dataset
from .models.base import ModelConfig
from .report import _agg, _cost_stats, _load_lines, _pct, result_path

_STYLE = """
:root {
  --bg: #f6f7f9; --card: #ffffff; --ink: #1c2128; --muted: #6b7280;
  --line: #e5e7eb; --accent: #2563eb; --ok: #16a34a; --bad: #dc2626;
  --warn-bg: #fff7ed;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#111418; --card:#1a1f26; --ink:#e6e8eb; --muted:#9aa3af;
          --line:#2a313a; --accent:#60a5fa; --ok:#4ade80; --bad:#f87171;
          --warn-bg:#2a2216; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Hiragino Sans GB",
        "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }
header.rpt h1 { font-size: 22px; margin: 0 0 4px; }
header.rpt .meta { color: var(--muted); font-size: 13px; }
header.rpt .meta code { background: var(--card); border: 1px solid var(--line);
  border-radius: 4px; padding: 1px 6px; }
h2 { font-size: 16px; margin: 36px 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid var(--line); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px; margin-top: 20px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 18px; }
.card .name { font-weight: 600; font-size: 14px; margin-bottom: 6px;
  display: flex; justify-content: space-between; gap: 8px; }
.badge { font-size: 11px; font-weight: 500; color: var(--muted);
  border: 1px solid var(--line); border-radius: 999px; padding: 1px 8px; }
.card .acc { font-size: 30px; font-weight: 700; letter-spacing: -0.5px; }
.card .sub { color: var(--muted); font-size: 12.5px; margin-top: 6px;
  display: flex; justify-content: space-between; }
.bar { height: 6px; border-radius: 3px; background: var(--line);
  margin: 10px 0 2px; overflow: hidden; }
.bar > i { display: block; height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--accent), var(--ok)); }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
  font-size: 13.5px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); }
th { background: var(--bg); font-weight: 600; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.heat-1 { background: hsl(8 80% 92%); } .heat-2 { background: hsl(35 80% 90%); }
.heat-3 { background: hsl(65 60% 90%); } .heat-4 { background: hsl(110 45% 90%); }
.heat-5 { background: hsl(135 50% 89%); }
@media (prefers-color-scheme: dark) {
  .heat-1 { background: hsl(8 45% 24%); } .heat-2 { background: hsl(35 40% 24%); }
  .heat-3 { background: hsl(65 32% 22%); } .heat-4 { background: hsl(110 28% 22%); }
  .heat-5 { background: hsl(135 32% 21%); }
}
details.case { background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; margin-bottom: 10px; overflow: hidden; }
details.case summary { cursor: pointer; padding: 10px 14px; font-size: 13.5px;
  display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
details.case[open] summary { border-bottom: 1px solid var(--line); }
details.case .body { padding: 12px 16px; font-size: 13.5px; }
.tag { font-size: 11px; border-radius: 999px; padding: 1px 8px;
  border: 1px solid var(--line); color: var(--muted); }
.tag.task { color: var(--accent); border-color: var(--accent); }
.kv { margin: 4px 0; }
.kv b { display: inline-block; min-width: 5em; color: var(--muted);
  font-weight: 500; }
code, pre { font-family: ui-monospace, "SF Mono", Consolas, Menlo, monospace; }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; white-space: pre-wrap; word-break: break-all;
  font-size: 12.5px; margin: 8px 0; }
.gold { color: var(--ok); font-weight: 600; }
.wrong { color: var(--bad); font-weight: 600; }
.note { color: var(--muted); font-size: 12.5px; margin-top: 10px; }
footer { margin-top: 48px; color: var(--muted); font-size: 12px;
  text-align: center; }
.warn { background: var(--warn-bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 14px; font-size: 13px; margin: 14px 0; }
"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _heat(pct_str: str) -> str:
    if pct_str == "-":
        return ""
    v = float(pct_str.rstrip("%"))
    cls = "heat-1" if v < 50 else "heat-2" if v < 70 else "heat-3" if v < 85 else \
          "heat-4" if v < 95 else "heat-5"
    return cls


def generate_html_report(
    dataset: Dataset,
    model_names: list[str],
    results_dir: Path,
    registry: dict[str, ModelConfig] | None = None,
    wrong_cases_max: int = 50,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    per_model = {
        n: _load_lines(result_path(n, dataset, results_dir)) for n in model_names
    }
    aggs = {n: _agg(lines) for n, lines in per_model.items()}
    costs = {n: _cost_stats(lines, (registry or {}).get(n)) for n, lines in per_model.items()}

    parts: list[str] = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>vlmeter 评测报告 — {_e(dataset.path.name)}</title>",
        f"<style>{_STYLE}</style></head><body><div class='wrap'>",
        "<header class='rpt'><h1>vlmeter 评测报告</h1><div class='meta'>",
        f"数据集 <code>{_e(dataset.path)}</code> · {len(dataset)} 图/题 / "
        f"{dataset.total_records} 个判定点 · 指纹 <code>{dataset.fingerprint[:8]}</code> · 生成于 {now}",
        "</div></header>",
    ]

    # ---- per-model cards ----
    parts.append("<div class='cards'>")
    for n in model_names:
        a, c = aggs[n], costs[n]
        acc = _pct(a["total_correct"], a["total_scored"])
        v = float(acc.rstrip("%")) if acc != "-" else 0
        per_1k = f"¥{c['per_call'] * 1000:.2f}/千次" if c["per_call"] is not None else "单价未配置"
        parts.append(
            f"<div class='card'><div class='name'><span>{_e(n)}</span>"
            f"<span class='badge'>{a['total_scored']}/{dataset.total_records} 判定点</span></div>"
            f"<div class='acc'>{acc}</div><div class='bar'><i style='width:{v:.1f}%'></i></div>"
            f"<div class='sub'><span>成本 {per_1k}</span><span>延迟 {a['avg_latency']}ms</span></div></div>"
        )
    parts.append("</div>")

    # ---- overview table ----
    parts.append("<h2>总览</h2><table><tr><th>模型</th><th class='num'>准确率</th>"
                 "<th class='num'>分类</th><th class='num'>检查项</th><th class='num'>VQA</th>"
                 "<th class='num'>解析失败</th><th class='num'>请求失败</th><th class='num'>未判定</th></tr>")
    for n in model_names:
        a = aggs[n]
        cls, chk, vqa = (a["by_task"].get(k, [0, 0]) for k in ("classification", "checklist", "vqa"))
        parts.append(
            f"<tr><td>{_e(n)}</td><td class='num'><b>{_pct(a['total_correct'], a['total_scored'])}</b></td>"
            f"<td class='num'>{_pct(*cls)}</td><td class='num'>{_pct(*chk)}</td><td class='num'>{_pct(*vqa)}</td>"
            f"<td class='num'>{a['parse_fail']}</td><td class='num'>{a['errors']}</td>"
            f"<td class='num'>{a['unscored']}</td></tr>"
        )
    parts.append("</table>")

    # ---- cost table ----
    parts.append("<h2>成本与性能</h2>")
    parts.append(
        "<table><tr><th>模型</th><th class='num'>准确率</th><th class='num'>调用次数</th>"
        "<th class='num'>tokens 入/出</th><th class='num'>本次总成本</th>"
        "<th class='num'>单次调用</th><th class='num'>每千次</th><th class='num'>平均延迟</th></tr>"
    )
    for n in model_names:
        a, c = aggs[n], costs[n]
        fmt = lambda v: f"¥{v:.4f}" if v is not None and v < 1 else (f"¥{v:.2f}" if v is not None else "未配置")
        per_1k = f"¥{c['per_call'] * 1000:.2f}" if c["per_call"] is not None else "-"
        parts.append(
            f"<tr><td>{_e(n)}</td><td class='num'>{_pct(a['total_correct'], a['total_scored'])}</td>"
            f"<td class='num'>{c['n_calls']}</td><td class='num'>{c['in_sum']}/{c['out_sum']}</td>"
            f"<td class='num'>{fmt(c['total'])}</td><td class='num'>{fmt(c['per_call'])}</td>"
            f"<td class='num'>{per_1k}</td><td class='num'>{a['avg_latency']}ms</td></tr>"
        )
    parts.append("</table><p class='note'>单价取自 vlmeter/config/models.yaml（请按官网价核对），"
                 "按实际 token 用量估算；judge 裁判费用另计。</p>")

    # ---- per-category heatmap ----
    cats = {it.category for it in dataset.items if it.task != "checklist"}
    for it in dataset.items:
        if it.task == "checklist":
            cats.update(ch.name for ch in it.checks)
    categories = sorted(cats)
    parts.append("<h2>分类目 / 检查项准确率</h2><table><tr><th>模型</th>")
    parts += [f"<th class='num'>{_e(c)}</th>" for c in categories]
    parts.append("</tr>")
    for n in model_names:
        a = aggs[n]
        parts.append(f"<tr><td>{_e(n)}</td>")
        for cat in categories:
            p = _pct(*a["by_cat"].get(cat, [0, 0]))
            parts.append(f"<td class='num {_heat(p)}'>{p}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    # ---- wrong cases ----
    parts.append("<h2>错误样本</h2>")
    shown = 0
    for n in model_names:
        for r in per_model[n]:
            if r.get("correct") is not False or shown >= wrong_cases_max:
                continue
            shown += 1
            reason = f" · {r['reason']}" if r.get("reason") else ""
            pred = r.get("prediction") if r["task"] != "checklist" else r.get("parsed")
            why = r.get("model_reason") or ""
            parts.append(
                f"<details class='case'><summary><b>{_e(n)}</b>"
                f"<span class='tag'>{_e(r['id'])}</span>"
                f"<span class='tag task'>{_e(r['task'])}</span>"
                f"<span class='tag'>{_e(r.get('category', ''))}</span></summary>"
                f"<div class='body'>"
                f"<div class='kv'><b>问题</b>{_e(r.get('question'))}{_e(reason)}</div>"
                f"<div class='kv'><b>标准答案</b><span class='gold'>{_e(r.get('gold'))}</span>"
                f"　<b>模型输出</b><span class='wrong'>{_e(pred)}</span></div>"
                + (f"<div class='kv'><b>模型依据</b>{_e(why)}</div>" if why else "")
                + f"<pre>{_e(r.get('prediction'))}</pre></div></details>"
            )
    if shown == 0:
        parts.append("<p>无答错样本 🎉</p>")

    parts.append(
        "<div class='warn'><b>统计口径</b>：准确率 = 判对判定点 / 已判定判定点；"
        "请求失败与未判定不计入分母；解析失败（模型不守格式且无法映射）按答错计。</div>"
        f"<footer>vlmeter v{__version__} · 准确率 × 成本计量流水线</footer>"
        "</div></body></html>"
    )
    return "".join(parts)


def html_report_path(dataset: Dataset, results_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"report_{dataset.path.stem}_{ts}.html"
