"""LLM-as-judge for open-ended VQA answers (text-only model call)."""
from __future__ import annotations

import json
import re

from .models.base import BaseVLModel
from .retry import retry_call

_JUDGE_PROMPT = """你是评测裁判。根据问题、标准答案和模型回答，判断模型回答是否正确。
判定标准：数值必须相等（单位不同但数值等价算对，如 1.5kg 与 1500g）；文字答案允许表述差异，但含义必须与标准答案一致；回答中包含正确答案且无矛盾信息即算对。

问题：{question}
标准答案：{gold}
模型回答：{prediction}

只输出一个 JSON 对象，格式：{{"correct": true}} 或 {{"correct": false}}，不要输出其他内容。"""


_CHOOSE_PROMPT = """你是评测裁判。模型本应从选项中选一个并回答字母，但它输出了一段描述。
请判断模型的回答最符合哪个选项。

问题：{question}
选项：
{options}
模型回答：{prediction}

只输出一个 JSON 对象，格式：{{"choice": "A"}}（字母为最符合的选项），不要输出其他内容。如果模型回答与任何选项都不符，输出 {{"choice": null}}。"""


class Judge:
    def __init__(self, model: BaseVLModel, attempts: int = 3):
        self.model = model
        self.attempts = attempts

    async def _ask(self, prompt: str) -> str | None:
        async def _call():
            return await self.model.chat(None, prompt)

        try:
            resp = await retry_call(_call, attempts=self.attempts)
        except Exception as e:  # noqa: BLE001
            return None
        m = re.search(r"\{[^{}]*\}", resp.text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    async def verify(self, question: str, gold: str, prediction: str) -> tuple[bool | None, str]:
        """Return (correct, reason). correct=None means judging failed."""
        prompt = _JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction[:500])
        verdict = await self._ask(prompt)
        if isinstance(verdict, dict) and isinstance(verdict.get("correct"), bool):
            return verdict["correct"], ""
        return None, "judge_parse_fail"

    async def choose(self, question: str, choices: list[str], prediction: str) -> int | None:
        """Map a free-form classification answer back to a choice index."""
        from .prompting import LETTERS

        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        prompt = _CHOOSE_PROMPT.format(
            question=question, options=options, prediction=prediction[:500]
        )
        verdict = await self._ask(prompt)
        if isinstance(verdict, dict):
            choice = verdict.get("choice")
            if isinstance(choice, str) and len(choice) == 1 and "A" <= choice.upper() <= "H":
                idx = ord(choice.upper()[0]) - 65
                if idx < len(choices):
                    return idx
        return None
