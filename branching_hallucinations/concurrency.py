"""Bounded async concurrency for experiment stages.

This is in-flight I/O (Azure, Serper, fetch), not OS threads. One
`concurrency` value caps how many work units run at once in a stage.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_CONCURRENCY = 8


def clamp_concurrency(value: int | None, default: int = DEFAULT_CONCURRENCY) -> int:
    if value is None:
        return default
    return max(1, int(value))


async def bounded_map(
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
) -> list[R]:
    """Run `fn` over `items` with at most `concurrency` in flight. Order preserved."""
    if not items:
        return []
    sem = asyncio.Semaphore(clamp_concurrency(concurrency))

    async def run(item: T) -> R:
        async with sem:
            return await fn(item)

    return list(await asyncio.gather(*[run(item) for item in items]))
