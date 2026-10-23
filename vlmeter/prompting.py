"""Prompt construction for each task type."""
from __future__ import annotations

from .dataset import Item

LETTERS = "ABCDEFGH"
YES, NO = "是", "否"


def build_prompt(item: Item) -> str:
    if item.task == "classification":
        lines = [item.question, "选项："]
        lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(item.choices)]
        lines.append("请只回答选项字母（如 A），不要输出其他内容。")
        return "\n".join(lines)
    if item.task == "checklist":
        lines = ["请逐条判断图片是否符合以下检查项。", "检查项："]
        for i, c in enumerate(item.checks):
            entry = f"{i + 1}. {c.name}"
            if c.criteria:
                entry += f"（判定标准：{c.criteria}）"
            lines.append(entry)
        lines.append(
            "只输出一个 JSON 对象，键为检查项编号，值为一个对象 "
            f'{{"verdict": "{YES}" 或 "{NO}", "reason": "一句话图片依据"}}，'
            f'如 {{\"1\": {{\"verdict\": \"{YES}\", \"reason\": \"图中左下角有一辆自行车\"}}}}，'
            "不要输出其他内容。"
        )
        return "\n".join(lines)
    return f"{item.question}\n请直接、简洁地回答，不要解释。"
