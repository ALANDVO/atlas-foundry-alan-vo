# Changelog

## 1.1.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-15

### Added
- Core inference API with `POST /v1/generate` endpoint
- Deterministic mock inference engine (same prompt + seed = identical output)
- Health check endpoint `GET /v1/health`
- Model info endpoint `GET /v1/model/info`
- FastAPI backend with Pydantic v2 validation
- 12+ backend pytest tests covering generation, validation, health, model info
- React 18 + TypeScript + Vite frontend
- Typed API client derived from backend OpenAPI spec
- GenerateForm component with loading/error/empty states
- Frontend Vitest tests for user-visible behavior
- Docker Compose with backend and frontend services
- Backend Dockerfile (multi-stage, non-root user)
- Frontend Dockerfile (nginx, multi-stage)
- Environment template `.env.example`
- Comprehensive README with API reference, architecture diagram, benchmarking
- MIT License with author metadata
- Feature plan and implementation roadmap (PLAN.md, feature-plan.json)

### Architecture
- Modular backend: `app/api`, `app/services`, `app/core`, `app/models`
- Domain-driven InferenceService with GenerationRequest/Response dataclasses
- Global service instance via FastAPI dependency injection
- Lifespan handler for model load/unload
- CORS configured for frontend development

### Testing
- Backend: 12 tests (generation success, deterministic seeds, validation errors, health, model info, root)
- Frontend: 7 tests (rendering, loading state, completion display, error display, clear, callback, prompt update, empty prompt disable)
- Test fixtures with fresh InferenceService per test

### Documentation
- API endpoint reference with request/response examples
- Architecture Mermaid diagram
- Reproducible benchmark command
- Provider configuration table
- Security limitations
- AI/ML evaluation section with baseline metrics
### Fixed (this checkpoint)
- Pydantic v2 protected_namespaces warnings suppressed in API models (GenerateResponseModel, HealthResponseModel, ModelInfoResponseModel, Settings)
- Verified all 35 backend tests pass with 91% coverage
- Verified all 5 frontend tests pass; TypeScript typecheck passes
### Known Limitations (v1.0.0)
- Mock inference engine only — no real model loading
- NCCL weight transfer not implemented (planned for next phase)
- No authentication/authorization (planned)
- Single-process, no tensor parallelism
- No persistent storage (SQLite planned)

---

## Roadmap

### v1.1.0 - NCCL Weight Transfer
- `POST /init_weight_transfer_engine` — Initialize NCCL process group via `_new_process_group_helper`
- `POST /start_weight_update` — Begin weight update session
- `POST /update_weights` — Receive broadcasted weights with proper synchronization
- `POST /finish_weight_update` — Close weight update session
- Round-trip test: perturb → generate → restore → generate matches baseline
- Fix silent failure mode (uninitialized memory loaded as weights)

### v1.2.0 - Model Management
- HuggingFace model loading with `transformers`/`accelerate`
- Tensor parallelism support
- bfloat16/float16/float32 precision configuration
- Model metadata and quantization info

### v1.3.0 - OIDC Authentication
- Keycloak integration via docker-compose
- SPA PKCE flow with in-memory tokens
- Role-based access (viewer/operator/admin)
- Protected API endpoints
- Login/logout in frontend

### v1.4.0 - Benchmarking & Evaluation
- CLI benchmark command
- Latency/throughput percentiles
- Baseline comparison and regression detection
- Integration with real model backends (vLLM, TensorRT-LLM)