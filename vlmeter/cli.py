"""Command-line interface: run / report / list-models."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import __version__
from .models import load_registry
from .runner import RESULTS_DIR, RunOptions, run_eval


def _load_dotenv() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--registry", type=Path, default=None, help="models.yaml 路径（默认包内 vlmeter/config/models.yaml）")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmeter", description="多模态大模型准确率评测流水线"
    )
    parser.add_argument("--version", action="version", version=f"vlmeter {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑评测（支持断点续跑）")
    run.add_argument("--models", required=True, help="逗号分隔的模型名，见 list-models")
    run.add_argument("--data", required=True, type=Path, help="数据集路径（.jsonl / .csv）")
    run.add_argument("--judge", default=None, help="judge 裁判模型名（VQA 语义判分）")
    run.add_argument("--limit", type=int, default=None, help="只评测前 N 题")
    run.add_argument("--fresh", action="store_true", help="忽略已缓存结果，全部重跑")
    _common_args(run)

    rep = sub.add_parser("report", help="由已有结果生成对比报告")
    rep.add_argument("--data", required=True, type=Path)
    rep.add_argument("--models", default=None, help="逗号分隔；缺省为有结果文件的所有注册模型")
    _common_args(rep)

    sub.add_parser("list-models", help="列出模型注册表").add_argument(
        "--registry", type=Path, default=None
    )
    return parser


def _resolve_model_names(args) -> list[str]:
    registry = load_registry(args.registry)
    if getattr(args, "models", None):
        names = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in names if m not in registry]
        if unknown:
            sys.exit(f"未知模型: {unknown}，可选: {sorted(registry)}")
        return names
    # report without --models: any registered model that has a result file
    from .dataset import load_dataset
    from .runner import result_path

    ds = load_dataset(args.data)
    names = [
        n for n in registry
        if result_path(n, ds, args.results_dir).exists()
    ]
    if not names:
        sys.exit("没有找到任何结果文件；请先运行 vlmeter run")
    return names


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    if args.command == "list-models":
        registry = load_registry(args.registry)
        print(f"{'名称':<28}{'provider':<12}{'模型ID':<30}{'单价(入/出 ¥/M)':<18}密钥环境变量")
        print("-" * 110)
        for name, cfg in registry.items():
            key = cfg.api_key_env or "-"
            price = (
                f"{cfg.price_input}/{cfg.price_output}" if cfg.price_input is not None else "-"
            )
            missing = (
                f"（未设置 {key}！）" if cfg.api_key_env and not os.environ.get(cfg.api_key_env)
                else ""
            )
            print(f"{name:<28}{cfg.provider:<12}{cfg.model:<30}{price:<18}{key}{missing}")
        return

    if args.command == "run":
        opts = RunOptions(
            models=[m.strip() for m in args.models.split(",") if m.strip()],
            data=args.data,
            judge=args.judge,
            limit=args.limit,
            fresh=args.fresh,
            registry_path=args.registry,
            results_dir=args.results_dir,
        )
        summaries = asyncio.run(run_eval(opts))
        print("\n运行完成：")
        for s in summaries:
            print(
                f"  {s.model}: 新作答 {s.answered}，跳过已缓存 {s.skipped_no_change}，"
                f"失败 {s.errors}，未判定 {s.unscored}"
            )
        # fall through to report generation below with resolved models
        args.models = ",".join(opts.models)

    from .dataset import load_dataset
    from .report import console_summary, generate_report, report_path

    registry = load_registry(args.registry)
    dataset = load_dataset(args.data)
    names = _resolve_model_names(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    console_summary(dataset, names, args.results_dir, registry)
    md = generate_report(dataset, names, args.results_dir, registry=registry)
    out = report_path(dataset, args.results_dir)
    out.write_text(md, encoding="utf-8")
    print(f"\n报告已生成: {out}")


if __name__ == "__main__":
    main()
