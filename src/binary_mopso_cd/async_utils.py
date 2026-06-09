from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")
_MISSING = object()


async def run_limited(
    items: Sequence[T],
    limit: int,
    worker: Callable[[T], Awaitable[R]],
) -> list[R]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    semaphore = asyncio.Semaphore(limit)
    results: list[R | object] = [_MISSING] * len(items)

    async def run_one(index: int, item: T) -> None:
        async with semaphore:
            results[index] = await worker(item)

    await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
    if any(result is _MISSING for result in results):
        raise RuntimeError("async worker did not produce all results")
    return [result for result in results]  # type: ignore[list-item]


def run_async(awaitable: Awaitable[R]) -> R:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Cannot run async task synchronously while an event loop is already running")
