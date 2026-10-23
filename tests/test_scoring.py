"""Scoring unit tests. Run: python tests/test_scoring.py (or pytest)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmeter.dataset import Item
from vlmeter.scoring import extract_numbers, normalize, parse_checklist, parse_choice, score


def _cls(answer: str, choices: list[str]) -> Item:
    return Item(id="t", image="x", task="classification", question="q",
                choices=choices, answer=answer)


def _vqa(answer: str, match: str) -> Item:
    return Item(id="t", image="x", task="vqa", question="q", answer=answer, match=match)


def test_parse_choice():
    ch = ["猫", "狗", "鸟"]
    assert parse_choice("A", ch) == 0
    assert parse_choice("B.", ch) == 1
    assert parse_choice("答案是 C", ch) == 2
    assert parse_choice("（A）", ch) == 0
    assert parse_choice("这是一只猫", ch) == 0
    assert parse_choice("我觉得是狗。", ch) == 1
    assert parse_choice("不知道", ch) is None
    # 模型枚举选项后再给结论：结论必须赢，不能取第一个字母
    assert parse_choice("A 为圆形，B 为方形。答案是B。", ch) == 1
    assert parse_choice("A. 猫 不符；B. 狗 符合。因此，答案是**B**", ch) == 1
    assert parse_choice("A. 猫: no; C. 鸟: yes, 故选C", ch) == 2
    assert parse_choice("answer is B", ch) == 1


def test_classification_score():
    item = _cls("圆形", ["圆形", "方形", "三角形"])
    assert score(item, "A").correct is True
    assert score(item, "三角形").correct is False
    assert score(item, "看不清").correct is False
    assert score(item, "看不清").reason == "parse_fail"
    # answer given as letter
    item2 = _cls("B", ["圆形", "方形", "三角形"])
    assert score(item2, "方形").correct is True


def test_vqa_match_modes():
    assert score(_vqa("3", "exact"), "3 个").correct is False
    assert score(_vqa("ZK-2049", "exact"), "zk2049").correct is True   # 忽略大小写/标点
    assert score(_vqa("ZK-2049", "exact"), "ZK-2048").correct is False
    assert score(_vqa("红色", "contains"), "图中车辆是红色的。").correct is True
    n = score(_vqa("3", "numeric"), "图中共有三个圆形")
    assert n.correct is True and n.match == "numeric"
    assert score(_vqa("12.5", "numeric"), "约 12.50").correct is True
    assert score(_vqa("3", "numeric"), "4").correct is False


def test_extract_numbers_chinese():
    assert extract_numbers("二十三个") == [23.0]
    assert extract_numbers("一共有5个") == [5.0]
    assert extract_numbers("十") == [10.0]


def test_normalize():
    assert normalize("  Hello, World! ") == "helloworld"
    assert normalize("３只") == normalize("3只")


def test_parse_checklist():
    # 提示词要求的格式：verdict + reason
    v = parse_checklist('{"1": {"verdict": "是", "reason": "左下角有自行车"}, '
                        '"2": {"verdict": "否", "reason": "台面有污渍"}}', 2)
    assert v == [(True, "左下角有自行车"), (False, "台面有污渍")]
    # 变体：数组、布尔、纯字符串（无理由）
    assert parse_checklist('{"1": ["是", "依据"], "2": ["否"]}', 2) == [(True, "依据"), (False, "")]
    assert parse_checklist('{"1": true, "2": false}', 2) == [(True, ""), (False, "")]
    assert parse_checklist('{"1": "是", "2": "否", "3": "是"}', 3) == [(True, ""), (False, ""), (True, "")]
    # JSON 前后有废话 / markdown 代码块
    assert parse_checklist('结论：```json\n{"1": {"verdict": "是", "reason": "可见"}}\n```', 1) == [(True, "可见")]
    # 漏答的检查项为 (None, "")
    assert parse_checklist('{"1": "是"}', 2) == [(True, ""), (None, "")]
    # 完全不是 JSON / 非法值
    assert parse_checklist("这张图很干净。", 2) == [(None, ""), (None, "")]
    assert parse_checklist('{"1": {"verdict": "也许", "reason": "看不清"}}', 1) == [(None, "看不清")]


def test_cost_estimate():
    from vlmeter.models.base import ModelConfig
    cfg = ModelConfig(name="t", provider="mock", model="m",
                      price_input=0.5, price_output=2.0)
    assert abs(cfg.estimate_cost(1_000_000, 1_000_000) - 2.5) < 1e-9
    assert abs(cfg.estimate_cost(1000, 500) - (0.5 * 1000 + 2.0 * 500) / 1e6) < 1e-12
    assert ModelConfig(name="t2", provider="mock", model="m").estimate_cost(1, 1) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} PASSED")
    print("all scoring tests passed")
