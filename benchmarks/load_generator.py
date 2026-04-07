"""
load_generator.py - Synthetic load generation for LLM Inference Server
MLOps Course - Milestone 5
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

UNIQUE_PROMPTS = [
    "Explain the concept of gradient descent in machine learning.",
    "What are the benefits of using Docker containers?",
    "Describe the CAP theorem in distributed systems.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how transformers work in natural language processing.",
    "What is the role of attention mechanisms in deep learning?",
    "Describe the architecture of a typical REST API.",
    "What are the trade-offs between SQL and NoSQL databases?",
    "Explain the concept of transfer learning.",
    "What is the purpose of regularization in machine learning?",
    "Describe how backpropagation works.",
    "What are the differences between batch and online learning?",
    "Explain the bias-variance trade-off.",
    "What is the purpose of a validation set in ML training?",
    "Describe the concept of feature engineering.",
    "What are embeddings and how are they used in NLP?",
    "Explain the concept of overfitting and how to prevent it.",
    "What is the role of the learning rate in training neural networks?",
    "Describe the differences between CNN and RNN architectures.",
    "What is a knowledge graph and how is it used?",
]

REPEATED_PROMPTS = [
    "What is machine learning?",
    "Explain artificial intelligence briefly.",
    "What is a neural network?",
    "Define deep learning.",
    "What is Python used for?",
]


def _get_prompts(n: int, repeat_fraction: float = 0.3) -> List[str]:
    prompts = []
    for _ in range(n):
        if random.random() < repeat_fraction:
            prompts.append(random.choice(REPEATED_PROMPTS))
        else:
            prompts.append(random.choice(UNIQUE_PROMPTS))
    return prompts


@dataclass
class RequestResult:
    prompt_length: int
    status_code: int
    latency_ms: float
    cached: bool
    error: Optional[str]
    timestamp: float


async def _send_request(session, base_url: str, prompt: str,
                        max_new_tokens: int, temperature: float) -> RequestResult:
    t0 = time.monotonic()
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": temperature}
    try:
        async with session.post(
            f"{base_url}/generate", json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            if resp.status == 200:
                data = await resp.json()
                return RequestResult(
                    prompt_length=len(prompt),
                    status_code=resp.status,
                    latency_ms=latency_ms,
                    cached=data.get("cached", False),
                    error=None,
                    timestamp=time.time(),
                )
            else:
                text = await resp.text()
                return RequestResult(
                    prompt_length=len(prompt),
                    status_code=resp.status,
                    latency_ms=latency_ms,
                    cached=False,
                    error=text[:200],
                    timestamp=time.time(),
                )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return RequestResult(
            prompt_length=len(prompt),
            status_code=0,
            latency_ms=latency_ms,
            cached=False,
            error=str(exc),
            timestamp=time.time(),
        )


async def run_load_test(base_url: str, rps: int, duration_seconds: int,
                        max_new_tokens: int = 32, temperature: float = 0.7,
                        repeat_fraction: float = 0.3) -> List[RequestResult]:
    interval = 1.0 / rps
    total_requests = rps * duration_seconds
    prompts = _get_prompts(total_requests, repeat_fraction)
    results = []
    connector = aiohttp.TCPConnector(limit=200)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(
                _send_request(session, base_url, prompt, max_new_tokens, temperature)
            )
            tasks.append(task)
            if (i + 1) < total_requests:
                await asyncio.sleep(interval)
        results = await asyncio.gather(*tasks)

    return list(results)


def summarize(results: List[RequestResult]) -> dict:
    import statistics
    latencies = [r.latency_ms for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]
    cached = [r for r in results if r.cached]

    if not latencies:
        return {"error": "No successful requests"}

    latencies.sort()

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return {
        "total_requests": len(results),
        "successful": len(latencies),
        "errors": len(errors),
        "error_rate": len(errors) / len(results),
        "cache_hits": len(cached),
        "cache_hit_rate": len(cached) / len(results),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p90_ms": percentile(latencies, 90),
        "latency_p99_ms": percentile(latencies, 99),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Load Generator")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=int, default=10)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repeat-fraction", type=float, default=0.3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Starting load test: {args.rps} RPS for {args.duration}s")
    results = asyncio.run(
        run_load_test(args.url, args.rps, args.duration,
                      args.max_new_tokens, args.temperature, args.repeat_fraction)
    )

    summary = summarize(results)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
        summary_path = out_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({**summary, "rps": args.rps, "duration": args.duration}, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()