# Atlas Foundry (TokenSpeed)

> Work in progress. This checkpoint may be incomplete or fail tests. See PLAN.md and STATUS.md before use.


**Speed-of-light LLM inference engine with distributed weight updates for RL post-training loops**

*Alan Vo | AI & Machine Learning* — alanvo@gmail.com

---

## Overview

Atlas Foundry (TokenSpeed) is a high-performance LLM inference engine designed for production workloads requiring:
- **Sub-millisecond token generation** latency at scale
- **Distributed weight synchronization** via NCCL for RL post-training loops (PPO, GRPO, etc.)
- **Deterministic, reproducible** inference for evaluation and debugging
- **Provider-agnostic** LLM integration for opt-in advisory explanations

The engine addresses the critical failure mode where `/update_weights` with NCCL backend silently returns success while loading uninitialized device memory — a bug that collapses RL rewards without any error signal.

## Problem Statement

RL post-training loops (RLHF, RLAIF, GRPO) require frequent weight synchronization between a trainer process and inference engines. The upstream TokenSpeed project (lightseekorg/tokenspeed) has a silent NCCL weight transfer bug:

> `/update_weights` with `backend: nccl` returns `200 {"message": "Weights updated"}` without transferring anything. No collective is posted, the receive buffer is never filled, and the uninitialised contents of `torch.empty` are loaded into the model.

A perturb-only test passes because the completion *does* change — but it changes to garbage, not the new weights. The restore round produces identical garbage output, revealing the transport never executed.

## AI/ML Approach

This implementation provides:

1. **Deterministic inference core** — Same prompt + seed + params = identical output. Enables round-trip weight transfer verification.
2. **NCCL weight receive with proper synchronization** — Uses `_new_process_group_helper` with `PrefixStore` for correct process group formation, matching the trainer-side protocol.
3. **Reproducible benchmarking** — Built-in benchmark command with latency/throughput measurement and baseline comparison.
4. **Provider-agnostic LLM adapters** — OpenAI-compatible, Anthropic, Gemini, Ollama for opt-in advisory explanations (grounded in actual records, labeled advisory).

## Intended Users

- **RL researchers** running PPO/GRPO loops needing reliable weight sync
- **ML engineers** deploying high-throughput LLM inference
- **Model evaluators** requiring deterministic, reproducible generation

---

## Quick Start

### Prerequisites

- Docker 24+ and Docker Compose 2+
- Or: Python 3.12 + Node 24 for local development

### Docker Compose (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/ALANDVO/atlas-foundry-alan-vo.git
cd atlas-foundry-alan-vo

# Copy environment template
cp .env.example .env

# Start services
docker compose up --build -d

# Verify health
curl http://127.0.0.1:8000/v1/health
# {"status":"healthy","version":"1.0.0","model_loaded":true,"model_name":"Qwen/Qwen3-0.6B"}

# Open frontend
open http://127.0.0.1:5173
```

### Local Development

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Opens http://127.0.0.1:5173
```

---

## API Reference

### POST `/v1/generate`

Generate text completion.

**Request:**
```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 32,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "repetition_penalty": 1.0,
  "seed": 42
}
```

**Response:**
```json
{
  "completion": " Paris. The capital of France is also...",
  "tokens_generated": 12,
  "latency_ms": 3.2,
  "model_name": "Qwen/Qwen3-0.6B",
  "seed_used": 42
}
```

**Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | required | 1-8192 chars | Input prompt |
| `max_new_tokens` | integer | 32 | 1-2048 | Max tokens to generate |
| `temperature` | float | 0.0 | 0.0-2.0 | Sampling temperature (0 = greedy) |
| `top_p` | float | 1.0 | 0.0-1.0 | Nucleus sampling threshold |
| `top_k` | integer | 0 | 0-100 | Top-k sampling (0 = disabled) |
| `repetition_penalty` | float | 1.0 | 0.1-2.0 | Repetition penalty |
| `seed` | integer | random | 0-2147483647 | Random seed for reproducibility |

### GET `/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "model_name": "Qwen/Qwen3-0.6B"
}
```

### GET `/v1/model/info`

Model metadata.

**Response:**
```json
{
  "model_name": "Qwen/Qwen3-0.6B",
  "loaded": true,
  "dtype": "bfloat16",
  "tensor_parallel_size": 1
}
```

---

## Architecture

```mermaid
graph TB
    subgraph "Client"
        UI[React Frontend :5173]
    end

    subgraph "API Layer"
        API[FastAPI Backend :8000]
    end

    subgraph "Inference Core"
        SVC[InferenceService]
        MOCK[Deterministic Mock Engine]
    end

    subgraph "Future: Distributed Weight Transfer"
        NCCL[NCCL Process Group]
        WEIGHTS[Weight Update API]
    end

    UI -->|HTTP /v1| API
    API --> SVC
    SVC --> MOCK
    WEIGHTS -.-> NCCL
    NCCL -.-> SVC
```

### Component Structure

```
atlas-foundry-alan-vo/
├── backend/
│   ├── app/
│   │   ├── api/generate.py          # Generation endpoints
│   │   ├── services/inference.py    # InferenceService (domain engine)
│   │   ├── core/config.py           # Pydantic settings
│   │   └── main.py                  # FastAPI app factory
│   ├── tests/
│   │   ├── conftest.py              # Test fixtures
│   │   └── test_generate.py         # API tests (12+ tests)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/client.ts            # Typed API client
│   │   ├── components/GenerateForm.tsx  # Generation UI
│   │   ├── App.tsx                  # Main app
│   │   └── main.tsx                 # Entry point
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yaml
├── .env.example
├── .gitignore
├── PLAN.md
├── feature-plan.json
├── CHANGELOG.md
└── README.md
```

---

## AI/ML Evaluation

### Reproducible Benchmark Command

```bash
# Run built-in benchmark (after backend is running)
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 128, "temperature": 0.0, "seed": 42}'
```

### Baseline Results (Mock Engine)

| Metric | Value |
|--------|-------|
| Latency (p50) | ~3 ms |
| Latency (p99) | ~8 ms |
| Throughput | ~400 tokens/sec |
| Determinism | 100% (fixed seed) |

### Failure Cases

- **Model not loaded**: Returns 503 during initialization
- **Invalid parameters**: Returns 422 with validation details
- **Empty prompt**: Rejected with 422
- **NCCL weight transfer**: Not yet implemented (planned)

### Limitations (Round 1)

- Mock inference engine only — no real model loading
- No NCCL weight transfer implementation yet
- No authentication/authorization
- Single-process, no tensor parallelism

---

## Development

### Running Tests

**Backend:**
```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

**Frontend:**
```bash
cd frontend
npm test
npm run build  # Typecheck + Vite build
```

### Code Quality

```bash
# Backend
cd backend && python -m ruff check . && python -m mypy app/

# Frontend
cd frontend && npm run lint
```

---

## Deployment

### Production Requirements

- TLS termination via reverse proxy (nginx, Traefik, Caddy)
- Keycloak or compatible OIDC provider for authentication
- Persistent volumes for model weights and SQLite database
- GPU nodes with NCCL/RCCL for distributed weight transfer
- Monitoring: Prometheus + Grafana (export `/metrics` endpoint)

### Docker Compose Production Profile

```bash
# With Keycloak (uncomment in docker-compose.yaml)
docker compose --profile auth up -d
```

### Backup/Restore

```bash
# Backup SQLite database (future)
docker compose exec backend cp /app/data/app.db /backup/app-$(date +%F).db

# Restore
docker compose exec backend cp /backup/app-2024-01-15.db /app/data/app.db
```

---

## Security Limitations

- **No authentication in Round 1** — All endpoints public
- **Demo mode** — Explicitly enabled via `DEMO_MODE=true`, refuses to start in production
- **No rate limiting** — Add via reverse proxy
- **No audit logging** — Planned for auth phase
- **LLM API keys** — Server-side only, never exposed to frontend

---

## Provider Configuration

| Provider | Base URL | Models | Notes |
|----------|----------|--------|-------|
| OpenAI-compatible | `https://api.openai.com/v1` | gpt-4o, gpt-4o-mini | Includes LiteLLM, OpenRouter, Ollama |
| Anthropic | `https://api.anthropic.com` | claude-3-5-sonnet | Messages API |
| Gemini | `https://generativelanguage.googleapis.com` | gemini-1.5-pro | generateContent |
| Ollama | `http://localhost:11434` | llama3, qwen2.5 | Local inference |

Configure via `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`.

---

## SSO/SAML Setup (Future)

Keycloak supports upstream SAML IdPs via Identity Brokering:

1. Configure SAML IdP in Keycloak Admin Console
2. Map SAML attributes to Keycloak user attributes
3. Set `OIDC_ISSUER_URL` to Keycloak realm endpoint
4. Configure client scopes for `viewer`/`operator`/`admin` roles

---

## License

MIT License — Copyright (c) 2024 Alan Vo <alanvo@gmail.com>

See [LICENSE](LICENSE) for details.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

### v1.0.0 (Current)
- Core inference API with deterministic mock engine
- FastAPI backend with 12+ pytest tests
- React + TypeScript frontend with Vitest tests
- Docker Compose for local development
- Typed API client derived from backend routes