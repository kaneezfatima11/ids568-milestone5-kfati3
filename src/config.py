"""
config.py - Configuration management for LLM Inference Server
MLOps Course - Milestone 5
"""

import os
from dataclasses import dataclass, field


@dataclass
class BatchingConfig:
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "8"))
    batch_timeout_ms: float = float(os.getenv("BATCH_TIMEOUT_MS", "50.0"))
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))


@dataclass
class CachingConfig:
    backend: str = os.getenv("CACHE_BACKEND", "memory")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    max_entries: int = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))


@dataclass
class ModelConfig:
    model_name: str = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")
    device: str = os.getenv("DEVICE", "cpu")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "64"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    do_sample: bool = os.getenv("DO_SAMPLE", "true").lower() == "true"


@dataclass
class ServerConfig:
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    workers: int = int(os.getenv("SERVER_WORKERS", "1"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


settings = ServerConfig()
