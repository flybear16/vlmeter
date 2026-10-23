"""Shared async retry helper (exponential backoff + jitter)."""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_call(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await fn()
        except retry_on as e:  # noqa: PERF203
            last = e
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2**i) + random.random())
    assert last is not None
    raise last


def short_err(e: BaseException) -> str:
    msg = str(e).replace("\n", " ")[:200]
    return f"{type(e).__name__}: {msg}"
