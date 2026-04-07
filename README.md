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
│   ├── server.py          # Main FastAPI inference server
│   ├── batching.py        # Dynamic batching logic
│   ├── caching.py         # Cache implementation
│   └── config.py          # Configuration management
├── benchmarks/
│   ├── run_benchmarks.py  # Full benchmark suite
│   ├── load_generator.py  # Async load generator
│   └── results/           # Raw benchmark data
├── analysis/
│   ├── performance_report.pdf   # Performance analysis
│   ├── governance_memo.pdf      # Governance memo
│   └── visualizations/          # Charts
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
python -m venv venv
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
| `BATCH_TIMEOUT_MS` | `50` | Batch flush timeout |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `CACHE_TTL_SECONDS` | `300` | Cache expiration time |
| `CACHE_MAX_ENTRIES` | `1000` | Max cache entries |

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
# Quick run (skips throughput tests)
python benchmarks/run_benchmarks.py --skip-throughput

# Full benchmark suite
python benchmarks/run_benchmarks.py --url http://localhost:8000 --rps-levels 10 50 100
```

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

# PII check (should return nothing)
grep -n "user_id\|user_name\|email\|username" src/caching.py || echo "✓ No PII found"
```

---

## Submission
```bash
git tag submission
git push --tags
```