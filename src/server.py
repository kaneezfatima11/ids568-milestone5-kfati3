"""
server.py - Main LLM Inference Server
MLOps Course - Milestone 5
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.batching import DynamicBatcher
from src.caching import InMemoryCache, build_cache, _make_cache_key
from src.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_name = settings.model.model_name
    device = settings.model.device
    logger.info("Loading model: %s on %s", model_name, device)

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(model_name)
    _model.to(device)
    _model.eval()
    logger.info("Model loaded successfully")


async def _infer_batch(prompts: list, max_new_tokens: int, temperature: float) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _infer_batch_sync, prompts, max_new_tokens, temperature)


def _infer_batch_sync(prompts: list, max_new_tokens: int, temperature: float) -> list:
    import torch
    device = settings.model.device
    inputs = _tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-6),
            do_sample=settings.model.do_sample,
            pad_token_id=_tokenizer.pad_token_id,
        )

    results = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[input_len:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(text.strip())
    return results


cache: Optional[InMemoryCache] = None
batcher: Optional[DynamicBatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache, batcher
    logger.info("=== Server starting up ===")

    _load_model()

    logger.info("Warming up model...")
    _infer_batch_sync(["Hello"], max_new_tokens=8, temperature=1.0)
    logger.info("Warm-up complete")

    cfg = settings.caching
    cache = build_cache(
        backend=cfg.backend,
        ttl_seconds=cfg.ttl_seconds,
        max_entries=cfg.max_entries,
        redis_host=cfg.redis_host,
        redis_port=cfg.redis_port,
        redis_db=cfg.redis_db,
    )

    bcfg = settings.batching
    batcher = DynamicBatcher(
        infer_fn=_infer_batch,
        max_batch_size=bcfg.max_batch_size,
        batch_timeout_ms=bcfg.batch_timeout_ms,
        max_queue_size=bcfg.max_queue_size,
    )
    await batcher.start()
    logger.info("=== Server ready ===")
    yield

    logger.info("=== Server shutting down ===")
    await batcher.stop()


app = FastAPI(
    title="LLM Inference Server",
    description="MLOps Milestone 5 — batching + caching inference API",
    version="1.0.0",
    lifespan=lifespan,
)


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_cache: bool = Field(default=True)


class InferenceResponse(BaseModel):
    generated_text: str
    cached: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    cache_backend: str


class MetricsResponse(BaseModel):
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    cache_size: int
    total_requests: int
    total_batches: int
    avg_batch_size: float
    avg_batch_wait_ms: float


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=settings.model.model_name,
        cache_backend=settings.caching.backend,
    )


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    t0 = time.monotonic()

    if request.use_cache:
        key = _make_cache_key(
            request.prompt,
            settings.model.model_name,
            request.max_new_tokens,
            request.temperature,
        )
        cached_result = await cache.get(key)
        if cached_result is not None:
            latency_ms = (time.monotonic() - t0) * 1000
            return InferenceResponse(generated_text=cached_result, cached=True, latency_ms=latency_ms)
    else:
        key = None

    try:
        result = await batcher.submit(request.prompt, request.max_new_tokens, request.temperature)
    except asyncio.QueueFull:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Queue full")
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    if request.use_cache and key:
        await cache.set(key, result)

    latency_ms = (time.monotonic() - t0) * 1000
    return InferenceResponse(generated_text=result, cached=False, latency_ms=latency_ms)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return MetricsResponse(
        cache_hits=cache.hits,
        cache_misses=cache.misses,
        cache_hit_rate=cache.hit_rate,
        cache_size=cache.size if hasattr(cache, "size") else -1,
        total_requests=batcher.total_requests,
        total_batches=batcher.total_batches,
        avg_batch_size=batcher.avg_batch_size,
        avg_batch_wait_ms=batcher.avg_wait_ms,
    )


@app.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache():
    await cache.clear()
    logger.info("Cache cleared via API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host=settings.host, port=settings.port, reload=False)