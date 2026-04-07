"""
run_benchmarks.py - Benchmark orchestration for LLM Inference Server
MLOps Course - Milestone 5
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

try:
    import aiohttp
except ImportError:
    aiohttp = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from benchmarks.load_generator import run_load_test, summarize, _send_request
except ImportError:
    run_load_test = summarize = _send_request = None

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def _clear_cache(session, base_url: str):
    try:
        async with session.delete(f"{base_url}/cache") as resp:
            if resp.status == 204:
                logger.info("Cache cleared")
    except Exception as e:
        logger.warning("Could not clear cache: %s", e)


async def _get_metrics(session, base_url: str) -> dict:
    try:
        async with session.get(f"{base_url}/metrics") as resp:
            return await resp.json()
    except Exception:
        return {}


async def _wait_for_server(base_url: str, timeout: int = 60):
    deadline = time.monotonic() + timeout
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    f"{base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as r:
                    if r.status == 200:
                        print(f"Server is up at {base_url}")
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"Server not ready after {timeout}s")


async def bench_single_vs_batch(base_url: str) -> dict:
    print("\n--- Benchmark 1: Single vs Batched Latency ---")
    connector = aiohttp.TCPConnector(limit=50)
    results = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        single_latencies = []
        for _ in range(10):
            r = await _send_request(session, base_url, "What is machine learning?", 32, 0.7)
            if r.status_code == 200:
                single_latencies.append(r.latency_ms)
        results["single_latency_ms"] = {
            "mean": sum(single_latencies) / len(single_latencies) if single_latencies else 0,
            "min": min(single_latencies) if single_latencies else 0,
            "max": max(single_latencies) if single_latencies else 0,
            "n": len(single_latencies),
        }
        print(f"  Single: mean={results['single_latency_ms']['mean']:.1f}ms")

        tasks = [
            _send_request(session, base_url, f"Explain concept number {i}.", 32, 0.7)
            for i in range(10)
        ]
        batch_results = await asyncio.gather(*tasks)
        batch_latencies = [r.latency_ms for r in batch_results if r.status_code == 200]
        results["batched_latency_ms"] = {
            "mean": sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0,
            "min": min(batch_latencies) if batch_latencies else 0,
            "max": max(batch_latencies) if batch_latencies else 0,
            "n": len(batch_latencies),
        }
        print(f"  Batched: mean={results['batched_latency_ms']['mean']:.1f}ms")

    return results


async def bench_cache(base_url: str) -> dict:
    print("\n--- Benchmark 2: Cold vs Warm Cache ---")
    connector = aiohttp.TCPConnector(limit=20)
    results = {}
    prompts = ["What is machine learning?", "Define deep learning.", "What is a neural network?"] * 5

    async with aiohttp.ClientSession(connector=connector) as session:
        await _clear_cache(session, base_url)
        cold_latencies = []
        for p in prompts[:5]:
            r = await _send_request(session, base_url, p, 32, 0.7)
            if r.status_code == 200:
                cold_latencies.append(r.latency_ms)
        results["cold_cache_latency_ms"] = {
            "mean": sum(cold_latencies) / len(cold_latencies) if cold_latencies else 0,
            "n": len(cold_latencies),
        }
        print(f"  Cold cache: mean={results['cold_cache_latency_ms']['mean']:.1f}ms")

        warm_latencies = []
        warm_hits = 0
        for p in prompts[:5]:
            r = await _send_request(session, base_url, p, 32, 0.7)
            if r.status_code == 200:
                warm_latencies.append(r.latency_ms)
                if r.cached:
                    warm_hits += 1
        results["warm_cache_latency_ms"] = {
            "mean": sum(warm_latencies) / len(warm_latencies) if warm_latencies else 0,
            "n": len(warm_latencies),
            "hits": warm_hits,
        }
        print(f"  Warm cache: mean={results['warm_cache_latency_ms']['mean']:.1f}ms, hits={warm_hits}/5")

    return results


async def bench_throughput(base_url: str, rps_levels: list) -> dict:
    print("\n--- Benchmark 3: Throughput vs Load ---")
    results = {}

    for rps in rps_levels:
        print(f"  Testing {rps} RPS for 20s...")
        tier_results = await run_load_test(
            base_url, rps=rps, duration_seconds=20,
            max_new_tokens=32, temperature=0.7, repeat_fraction=0.3
        )
        summary = summarize(tier_results)
        summary["rps_target"] = rps
        results[f"rps_{rps}"] = summary
        print(f"    P50={summary.get('latency_p50_ms', 0):.0f}ms "
              f"P99={summary.get('latency_p99_ms', 0):.0f}ms "
              f"errors={summary.get('error_rate', 0):.1%}")

        import csv
        from dataclasses import asdict
        csv_path = RESULTS_DIR / f"throughput_rps{rps}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(tier_results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(r) for r in tier_results)

    return results


async def bench_cache_hitrate(base_url: str) -> dict:
    print("\n--- Benchmark 4: Cache Hit-Rate Over Time ---")
    connector = aiohttp.TCPConnector(limit=20)
    snapshots = []

    from benchmarks.load_generator import REPEATED_PROMPTS, UNIQUE_PROMPTS
    import random

    async with aiohttp.ClientSession(connector=connector) as session:
        await _clear_cache(session, base_url)
        for batch_num in range(10):
            prompts = [
                random.choice(REPEATED_PROMPTS if random.random() < 0.4 else UNIQUE_PROMPTS)
                for _ in range(5)
            ]
            tasks = [_send_request(session, base_url, p, 32, 0.7) for p in prompts]
            await asyncio.gather(*tasks)
            metrics = await _get_metrics(session, base_url)
            snapshots.append({
                "batch": batch_num + 1,
                "requests_so_far": (batch_num + 1) * 5,
                "cache_hits": metrics.get("cache_hits", 0),
                "cache_misses": metrics.get("cache_misses", 0),
                "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                "cache_size": metrics.get("cache_size", 0),
            })
            print(f"  Batch {batch_num+1}: hit_rate={metrics.get('cache_hit_rate', 0):.1%} "
                  f"size={metrics.get('cache_size', 0)}")

    return {"snapshots": snapshots}


async def run_all(base_url: str, rps_levels: list, skip_throughput: bool):
    await _wait_for_server(base_url)
    all_results = {}
    all_results["single_vs_batch"] = await bench_single_vs_batch(base_url)
    all_results["cache_comparison"] = await bench_cache(base_url)
    if not skip_throughput:
        all_results["throughput"] = await bench_throughput(base_url, rps_levels)
    all_results["cache_hitrate"] = await bench_cache_hitrate(base_url)

    out_path = RESULTS_DIR / "benchmark_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== All results saved to {out_path} ===")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Suite")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps-levels", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--skip-throughput", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_all(args.url, args.rps_levels, args.skip_throughput))


if __name__ == "__main__":
    main()