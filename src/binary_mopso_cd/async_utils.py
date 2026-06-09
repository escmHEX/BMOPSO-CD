from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


async def run_limited(
    items: list[T],
    max_concurrent: int,
    worker: Callable[[T], Awaitable[R]],
) -> list[R]:
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return list(await asyncio.gather(*(guarded(item) for item in items)))


def run_async(awaitable: Awaitable[R]) -> R:
    return asyncio.run(awaitable)
