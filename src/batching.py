"""
batching.py - Dynamic request batching for LLM Inference Server
MLOps Course - Milestone 5

Strategy: Hybrid (size OR timeout) — whichever comes first.
  - Collects concurrent requests into a queue.
  - Flushes when max_batch_size is reached OR batch_timeout_ms elapses.
  - Uses asyncio.Lock and asyncio.Event to prevent race conditions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    prompt: str
    max_new_tokens: int
    temperature: float
    future: asyncio.Future = field(default_factory=asyncio.Future)
    enqueue_time: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    def __init__(
        self,
        infer_fn: Callable[[List[str], int, float], Coroutine[Any, Any, List[str]]],
        max_batch_size: int = 8,
        batch_timeout_ms: float = 50.0,
        max_queue_size: int = 100,
    ):
        self.infer_fn = infer_fn
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_wait_ms: float = 0.0

    async def start(self) -> None:
        self._stop_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(
            "DynamicBatcher started (max_batch=%d, timeout=%.1f ms)",
            self.max_batch_size, self.batch_timeout_ms,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._worker_task:
            await self._worker_task

    async def submit(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        req = BatchRequest(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        await self._queue.put(req)
        self.total_requests += 1
        return await req.future

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            batch: List[BatchRequest] = []
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            batch_start = time.monotonic()
            deadline = batch_start + self.batch_timeout_ms / 1000.0

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            wait_ms = (time.monotonic() - batch_start) * 1000
            self.total_batch_wait_ms += wait_ms
            self.total_batches += 1

            logger.info("Flushing batch: size=%d, wait=%.1f ms", len(batch), wait_ms)

            prompts = [r.prompt for r in batch]
            max_tokens = batch[0].max_new_tokens
            temperature = batch[0].temperature

            try:
                results = await self.infer_fn(prompts, max_tokens, temperature)
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)
            except Exception as exc:
                logger.exception("Batch inference error: %s", exc)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    @property
    def avg_batch_size(self) -> float:
        return self.total_requests / self.total_batches if self.total_batches > 0 else 0.0

    @property
    def avg_wait_ms(self) -> float:
        return self.total_batch_wait_ms / self.total_batches if self.total_batches > 0 else 0.0