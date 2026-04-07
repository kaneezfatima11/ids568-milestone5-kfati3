"""
caching.py - Cache implementation for LLM Inference Server
MLOps Course - Milestone 5

Privacy-preserving design:
  - Cache keys are SHA-256 hashes of (prompt + model params) — no PII stored.
  - No personal identifiers are ever persisted in keys or values.
  - TTL-based expiration prevents stale/sensitive data from lingering.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


def _make_cache_key(prompt: str, model_name: str, max_new_tokens: int, temperature: float) -> str:
    payload = json.dumps(
        {"prompt": prompt, "model": model_name, "max_new_tokens": max_new_tokens, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class InMemoryCache:
    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 300):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            expires_at = time.monotonic() + self.ttl_seconds
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class RedisCache:
    def __init__(self, host: str, port: int, db: int, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.Redis(host=host, port=port, db=db, decode_responses=True)
        except ImportError:
            self._client = None

    async def get(self, key: str) -> Optional[str]:
        if self._client is None:
            return None
        value = await self._client.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, key: str, value: str) -> None:
        if self._client is None:
            return
        await self._client.setex(key, self.ttl_seconds, value)

    async def clear(self) -> None:
        if self._client is None:
            return
        await self._client.flushdb()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def build_cache(backend: str, ttl_seconds: int, max_entries: int,
                redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
    if backend == "redis":
        return RedisCache(host=redis_host, port=redis_port, db=redis_db, ttl_seconds=ttl_seconds)
    return InMemoryCache(max_entries=max_entries, ttl_seconds=ttl_seconds)
