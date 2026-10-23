"""Generate a small synthetic sample dataset (15 items) with known ground truth.

Covers both supported task types so the whole pipeline — classification option
matching, numeric/exact VQA scoring, reporting — can be verified offline.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
SHAPES = ("圆形", "方形", "三角形")
COLORS = {"红色": (200, 40, 40), "蓝色": (40, 80, 200), "绿色": (40, 160, 60), "黄色": (220, 180, 30)}


def _canvas() -> Image.Image:
    return Image.new("RGB", (256, 256), (250, 250, 250))


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: tuple, cx: int, cy: int, r: int) -> None:
    box = (cx - r, cy - r, cx + r, cy + r)
    if shape == "圆形":
        draw.ellipse(box, fill=color)
    elif shape == "方形":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def main() -> None:
    random.seed(42)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    def save(img: Image.Image, name: str) -> str:
        rel = f"images/{name}.png"
        img.save(IMG_DIR / f"{name}.png")
        return rel

    # 1) 形状分类 x6
    for i in range(6):
        shape = SHAPES[i % 3]
        color = random.choice(list(COLORS.values()))
        img, d = _canvas(), None
        d = ImageDraw.Draw(img)
        _draw_shape(d, shape, color, 128, 128, 60)
        items.append({
            "id": f"cls_shape_{i:02d}",
            "image": save(img, f"cls_shape_{i:02d}"),
            "task": "classification",
            "question": "图中是什么形状？",
            "choices": list(SHAPES),
            "answer": shape,
            "category": "形状分类",
        })

    # 2) 颜色分类 x3
    for i, (cname, rgb) in enumerate(list(COLORS.items())[:3]):
        img = _canvas()
        d = ImageDraw.Draw(img)
        _draw_shape(d, random.choice(SHAPES), rgb, 128, 128, 60)
        items.append({
            "id": f"cls_color_{i:02d}",
            "image": save(img, f"cls_color_{i:02d}"),
            "task": "classification",
            "question": "图中图形的主要颜色是什么？",
            "choices": list(COLORS.keys()),
            "answer": cname,
            "category": "颜色分类",
        })

    # 3) 计数 VQA x3（numeric 判分）
    for i, n in enumerate([2, 3, 5]):
        img = _canvas()
        d = ImageDraw.Draw(img)
        for k in range(n):
            x = 40 + k * (176 // max(n - 1, 1)) if n > 1 else 128
            _draw_shape(d, "圆形", (60, 90, 200), x, 128, 22)
        items.append({
            "id": f"vqa_count_{i:02d}",
            "image": save(img, f"vqa_count_{i:02d}"),
            "task": "vqa",
            "question": f"图中一共有几个圆形？只回答数字。",
            "answer": str(n),
            "match": "numeric",
            "category": "计数",
        })

    # 4) 编号 OCR VQA x3（exact 判分）
    for i, code in enumerate(["ZK-2049", "AB-7781", "MX-0357"]):
        img = _canvas()
        d = ImageDraw.Draw(img)
        d.rectangle((28, 88, 228, 168), outline=(40, 40, 40), width=3)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except OSError:
            font = ImageFont.load_default()
        d.text((48, 108), code, fill=(20, 20, 20), font=font)
        items.append({
            "id": f"vqa_ocr_{i:02d}",
            "image": save(img, f"vqa_ocr_{i:02d}"),
            "task": "vqa",
            "question": "图中方框里写的编号是什么？只回答编号本身。",
            "answer": code,
            "match": "exact",
            "category": "文字识别",
        })

    # 5) 场景检查 checklist x3（一图多检查项，单次调用）
    scenes = [
        (  # 蓝色圆形 + 红色方形
            [("圆形", COLORS["蓝色"], 60, 128), ("方形", COLORS["红色"], 190, 128)],
            [("图中存在圆形", True, "可见完整的圆形轮廓，部分遮挡也算"),
             ("图中存在方形", True, "可见完整的方形轮廓，部分遮挡也算"),
             ("所有图形都是蓝色", False, "图中每一个图形均为蓝色才算符合")],
        ),
        (  # 两个绿色三角形
            [("三角形", COLORS["绿色"], 70, 128), ("三角形", COLORS["绿色"], 180, 128)],
            [("图中存在三角形", True, "可见三角形轮廓，部分遮挡也算"),
             ("图中存在圆形", False, "可见完整的圆形轮廓才算符合"),
             ("图中有三个以上图形", False, "图形总数 ≥ 3 才算符合")],
        ),
        (  # 单个黄色方形
            [("方形", COLORS["黄色"], 128, 128)],
            [("图中存在方形", True, "可见完整的方形轮廓，部分遮挡也算"),
             ("图形是黄色", True, "主体图形的主色为黄色"),
             ("图中存在圆形", False, "可见完整的圆形轮廓才算符合")],
        ),
    ]
    for i, (shapes_spec, checks) in enumerate(scenes):
        img = _canvas()
        d = ImageDraw.Draw(img)
        for shape, color, cx, cy in shapes_spec:
            _draw_shape(d, shape, color, cx, cy, 45)
        items.append({
            "id": f"chk_scene_{i:02d}",
            "image": save(img, f"chk_scene_{i:02d}"),
            "task": "checklist",
            "checks": [{"name": n, "answer": a, "criteria": c} for n, a, c in checks],
        })

    out = ROOT / "data" / "sample.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"生成 {len(items)} 条数据 -> {out}")


if __name__ == "__main__":
    main()
