# Mini AdTech Engine

> **Production version:** [`openrtb-semantic-matcher`](https://github.com/zizimind/openrtb-semantic-matcher) — the full Go system handling 250+ QPS with OpenRTB-compliant bidding and semantic matching.
> This repo is the educational companion: same core concepts, 960 lines of Python.

A compact, domain-agnostic ranking and decision engine implemented in pure Python.

The same retrieval + scoring pipeline is reused across:
- ad auctions,
- health plan recommendation,
- content/product ranking.

## What is included

- `mini_adtech.py`: retrieval, ranking models, auction/recommendation logic, pacing, caps, throttling, RAG enrichment, and simulations.
- `tests/test_mini_adtech.py`: unit tests, including edge cases.
- `benchmark.py`: deterministic benchmark runner writing `artifacts/benchmark.json`.

## Requirements

- Python 3.10+

## Local setup with custom virtualenv

You can choose any virtualenv directory name/location.

Example with a custom path:

```bash
python3 -m venv .venv-mini
source .venv-mini/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependency:
- `requirements.txt` (tests + core project)

Optional RAG dependency pack:
- `requirements-rag.txt` (LangChain/Ollama/FAISS stack)

You can also use `Makefile` targets with a custom venv:

```bash
make install VENV_DIR=.venv-mini
make install-rag VENV_DIR=.venv-mini
make test VENV_DIR=.venv-mini
make run VENV_DIR=.venv-mini
make benchmark VENV_DIR=.venv-mini
```

## Run the simulations

```bash
python mini_adtech.py
```

This runs:
- Simulation A: ad auction flow.
- Simulation B: health recommendation flow.
- Simulation C: RAG enrichment flow.

## Run tests

```bash
pytest -q
```

## Benchmark artifact (evidence-ready)

Generate deterministic KPI output:

```bash
python benchmark.py
```

or

```bash
make benchmark VENV_DIR=.venv-mini
```

This writes:
- `artifacts/benchmark.json`

The artifact includes:
- adtech: no-fill rate, total revenue, RPM, p50/p95 latency
- health: enrollment rate, average top-1 fit score, p50/p95 latency

## Docker option (dockerized workflow)

Build image:

```bash
docker build -t mini-adtech:latest .
```

Run simulations in container:

```bash
docker run --rm mini-adtech:latest
```

Run tests in container:

```bash
docker run --rm mini-adtech:latest pytest -q
```

Equivalent `Makefile` commands:

```bash
make docker-build
make docker-run
make docker-test
```

## Docker Compose option

Run simulations:

```bash
docker compose run --rm app
```

Run tests:

```bash
docker compose run --rm test
```

Equivalent `Makefile` commands:

```bash
make compose-run
make compose-test
```

## Engineering notes

- The core engine is intentionally dependency-light and deterministic enough for unit testing.
- Input validation is added for key edge cases (negative budgets, negative token costs, invalid reserve CPM, etc.).
- Recommendation and auction stages share the same candidate generation pipeline, but differ at final decision logic.

## Known limitations (explicit)

- Uses synthetic, in-repo datasets; no production data or PHI.
- Single-process simulation model; no networked serving path in this repo.
- Metrics are simulation-derived and intended for architecture validation, not business forecasting.
- No explicit fairness, calibration, or drift-monitoring module yet.
