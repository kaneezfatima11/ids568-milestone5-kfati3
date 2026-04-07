# ids568-milestone5-fatim

**MLOps Course — Milestone 5: LLM Inference Optimization with Batching and Caching**

---

## Overview

This project implements a production-ready LLM inference API with:
- **Dynamic request batching** — hybrid size/timeout strategy to maximize GPU utilization
- **Intelligent response caching** — SHA-256 keyed, TTL-based, privacy-preserving (no PII stored)
- **Concurrent request safety** — asyncio primitives throughout (no race conditions)
- **Full benchmark suite** — latency, throughput, cache hit-rate, cold vs. warm comparisons

---

## Repository Structure
```
ids568-milestone5-fatim/
├── src/
│   ├── __init__.py
│   ├── server.py          # Main FastAPI inference server
│   ├── batching.py        # Dynamic batching logic
│   ├── caching.py         # Cache implementation
│   └── config.py          # Configuration management
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmarks.py  # Full benchmark suite
│   ├── load_generator.py  # Async load generator
│   └── results/
│       ├── benchmark_summary.json
│       ├── load_test_rps10.csv
│       ├── load_test_rps50.csv
│       ├── load_test_rps100.csv
│       ├── cold_cache_test.csv
│       └── warm_cache_test.csv
├── analysis/
│   ├── performance_report.pdf
│   ├── governance_memo.pdf
│   └── visualizations/
│       ├── chart1_single_vs_batch.png
│       ├── chart2_cache_performance.png
│       ├── chart3_throughput_load.png
│       └── chart4_cache_hit_rate_over_time.png
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- GPU with CUDA (optional but recommended)
- Redis (optional, for distributed caching)

### Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

All settings are via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `sshleifer/tiny-gpt2` | HuggingFace model |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `MAX_NEW_TOKENS` | `64` | Max tokens generated |
| `MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `BATCH_TIMEOUT_MS` | `50` | Batch flush timeout (ms) |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `CACHE_TTL_SECONDS` | `300` | Cache expiration time |
| `CACHE_MAX_ENTRIES` | `1000` | Max cache entries (LRU) |

---

## Running the Server
```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Verify it works:
```bash
curl http://localhost:8000/health
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/generate` | Generate text |
| `GET` | `/metrics` | Cache and batch metrics |
| `DELETE` | `/cache` | Clear cache |

Example request:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_new_tokens": 64}'
```

---

## Running Benchmarks
```bash
# Low load - 10 RPS for 20 seconds
python3 benchmarks/load_generator.py \
  --url http://localhost:8000 \
  --rps 10 --duration 20 \
  --output benchmarks/results/load_test_rps10.csv

# Medium load - 50 RPS for 20 seconds
python3 benchmarks/load_generator.py \
  --url http://localhost:8000 \
  --rps 50 --duration 20 \
  --output benchmarks/results/load_test_rps50.csv

# High load - 100 RPS for 20 seconds
python3 benchmarks/load_generator.py \
  --url http://localhost:8000 \
  --rps 100 --duration 20 \
  --output benchmarks/results/load_test_rps100.csv

# Cold cache test
curl -X DELETE http://localhost:8000/cache
python3 benchmarks/load_generator.py \
  --url http://localhost:8000 \
  --rps 10 --duration 10 \
  --repeat-fraction 0.5 \
  --output benchmarks/results/cold_cache_test.csv

# Warm cache test (run immediately after cold)
python3 benchmarks/load_generator.py \
  --url http://localhost:8000 \
  --rps 10 --duration 10 \
  --repeat-fraction 0.5 \
  --output benchmarks/results/warm_cache_test.csv
```

---

## Real Benchmark Results

### Cold vs Warm Cache
| Metric | Cold Cache | Warm Cache |
|---|---|---|
| Total Requests | 100 | 100 |
| Cache Hit Rate | 74% | 98% |
| Mean Latency | 70.72ms | 5.14ms |
| P50 Latency | 1.54ms | 1.47ms |
| P90 Latency | 290.84ms | 1.65ms |
| P99 Latency | 509.69ms | 186.07ms |

### Throughput at Multiple Load Levels
| Load | RPS | Requests | Cache Hit Rate | Mean Latency | P99 Latency |
|---|---|---|---|---|---|
| Low | 10 | 200 | 87% | 37.88ms | 471.39ms |
| Medium | 50 | 1000 | 100% | 1.30ms | 1.72ms |
| High | 100 | 2000 | 100% | 1.12ms | 1.55ms |

### Key Findings
- Warm cache reduces mean latency from **70.72ms to 5.14ms** — a **13.8x speedup**
- At 100 RPS with warm cache, mean latency is only **1.12ms**
- Cache hit rate reaches **98-100%** at medium and high load levels
- Zero errors across all load levels demonstrating system stability

---

## Sanity Checks
```bash
# File existence
test -f src/server.py && echo "✓ server.py" || echo "✗ MISSING"
test -f src/batching.py && echo "✓ batching.py" || echo "✗ MISSING"
test -f src/caching.py && echo "✓ caching.py" || echo "✗ MISSING"
test -f src/config.py && echo "✓ config.py" || echo "✗ MISSING"

# Syntax checks
python -m py_compile src/server.py && echo "✓ server.py OK"
python -m py_compile src/batching.py && echo "✓ batching.py OK"
python -m py_compile src/caching.py && echo "✓ caching.py OK"
python -m py_compile src/config.py && echo "✓ config.py OK"

# PII check
grep -n "user_id\|user_name\|email\|username" src/caching.py || echo "✓ No PII found"
```

---

## Design Decisions

### Batching Strategy: Hybrid (Size OR Timeout)
Flushes when MAX_BATCH_SIZE requests accumulate OR BATCH_TIMEOUT_MS elapses — whichever comes first. Ensures high-load batches fill quickly while low-load requests are not held indefinitely.

### Cache Key Design: Privacy-First
```
key = SHA256(json({"prompt": p, "model": m, "max_new_tokens": n, "temperature": t}))
```
No user identifiers ever appear in keys or values. Deterministic and one-way.

### Concurrency: asyncio Throughout
All shared state protected by asyncio.Lock. Model inference runs in thread pool via run_in_executor() to avoid blocking the event loop.

---

## Submission
```bash
git tag submission
git push --tags
```