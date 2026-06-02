from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, enabled: bool, outdir: Path, directory_name: str, interval: int):
        self.enabled = enabled
        self.outdir = outdir
        self.directory_name = directory_name
        self.interval = interval
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._pending: Future[Any] | None = None

    def submit(self, generation: int, payload: dict[str, Any]) -> None:
        if not self.enabled or generation % self.interval != 0:
            return
        self._wait_pending()
        target_dir = self.outdir / self.directory_name
        target = target_dir / f"generation_{generation:04d}.json"
        self._pending = self._executor.submit(self._write_atomic, target, payload) if self._executor else None

    def close(self) -> None:
        self._wait_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _wait_pending(self) -> None:
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    @staticmethod
    def _write_atomic(target: Path, payload: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(target)
