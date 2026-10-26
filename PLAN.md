# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

core-inference-api

## Roadmap

### core-inference-api — Core Inference API - Text Generation Endpoint

Status: done

Acceptance:
- POST /generate accepts prompt and sampling params, returns completion
- Health check endpoint /health returns service status
- Backend runs with pytest tests passing
- Frontend builds and typechecks
- Docker compose starts backend and frontend

Implementation: `backend/app/main.py`, `backend/app/api/generate.py`, `backend/app/services/inference.py`, `backend/requirements.txt`, `backend/tests/test_generate.py`, `frontend/package.json`, `frontend/src/api/client.ts`, `frontend/src/components/GenerateForm.tsx`, `frontend/src/App.tsx`, `docker-compose.yaml`, `Dockerfile.backend`, `Dockerfile.frontend`

Tests: `backend/tests/test_generate.py`, `frontend/src/components/GenerateForm.test.tsx`

### weight-transfer-nccl — NCCL Weight Transfer for Distributed Training

Status: planned

Acceptance:
- POST /init_weight_transfer_engine initializes NCCL process group
- POST /start_weight_update begins weight update session
- POST /update_weights receives broadcasted weights via NCCL
- POST /finish_weight_update closes weight update session
- Round-trip test: perturb weights -> generate -> restore weights -> generate matches baseline

### model-management — Model Loading and Management

Status: planned

Acceptance:
- Load HuggingFace models with configurable tensor parallelism
- Model metadata endpoint
- Support for bfloat16/float16 precision

### auth-oidc — OIDC Authentication with Keycloak

Status: planned

Acceptance:
- Keycloak integration via docker-compose
- Protected API endpoints with role-based access
- Login/logout flow in frontend

### benchmarks-evaluation — Benchmarking and Evaluation Harness

Status: planned

Acceptance:
- Reproducible benchmark command
- Latency/throughput metrics
- Comparison baseline storage

