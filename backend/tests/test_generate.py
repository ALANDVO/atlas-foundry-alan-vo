"""
Tests for the generation API endpoint.

Covers:
- Successful generation with valid parameters
- Validation errors for invalid inputs
- Empty prompt rejection
- Deterministic output with fixed seed
- Health check endpoint
- Model info endpoint
"""

import pytest
from fastapi.testclient import TestClient


class TestGenerateEndpoint:
    """Tests for POST /v1/generate."""

    def test_generate_success(self, client: TestClient):
        """Successful generation returns completion with metadata."""
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "The capital of France is",
                "max_new_tokens": 16,
                "temperature": 0.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "completion" in data
        assert isinstance(data["completion"], str)
        assert len(data["completion"]) > 0
        assert data["tokens_generated"] > 0
        assert data["latency_ms"] > 0
        assert data["model_name"] == "test-model"
        assert data["seed_used"] is not None

    def test_generate_deterministic_with_seed(self, client: TestClient):
        """Same prompt + seed + params produces identical output."""
        prompt = "Hello world"
        params = {
            "prompt": prompt,
            "max_new_tokens": 10,
            "temperature": 0.0,
            "seed": 42,
        }

        response1 = client.post("/v1/generate", json=params)
        response2 = client.post("/v1/generate", json=params)

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["completion"] == response2.json()["completion"]
        assert response1.json()["seed_used"] == response2.json()["seed_used"] == 42

    def test_generate_different_seeds_different_output(self, client: TestClient):
        """Different seeds produce different completions (with high probability)."""
        prompt = "Test prompt for seed variation"
        base_params = {"prompt": prompt, "max_new_tokens": 8, "temperature": 0.0}

        response1 = client.post("/v1/generate", json={**base_params, "seed": 1})
        response2 = client.post("/v1/generate", json={**base_params, "seed": 2})

        assert response1.status_code == 200
        assert response2.status_code == 200
        # With different seeds, completions should differ (very high probability)
        assert response1.json()["completion"] != response2.json()["completion"]
        assert response1.json()["seed_used"] == 1
        assert response2.json()["seed_used"] == 2

    def test_generate_temperature_zero_greedy(self, client: TestClient):
        """Temperature 0 produces deterministic greedy output."""
        params = {
            "prompt": "Deterministic test",
            "max_new_tokens": 10,
            "temperature": 0.0,
            "seed": 123,
        }

        r1 = client.post("/v1/generate", json=params)
        r2 = client.post("/v1/generate", json=params)

        assert r1.json()["completion"] == r2.json()["completion"]

    def test_generate_validation_empty_prompt_rejected(self, client: TestClient):
        """Empty prompt returns 422 validation error."""
        response = client.post("/v1/generate", json={"prompt": ""})
        assert response.status_code == 422

    def test_generate_validation_whitespace_prompt_rejected(self, client: TestClient):
        """Whitespace-only prompt returns 422 validation error."""
        response = client.post("/v1/generate", json={"prompt": "   \n\t  "})
        assert response.status_code == 422

    def test_generate_validation_max_tokens_too_high(self, client: TestClient):
        """max_new_tokens > 2048 returns 422."""
        response = client.post("/v1/generate", json={"prompt": "test", "max_new_tokens": 3000})
        assert response.status_code == 422

    def test_generate_validation_max_tokens_zero(self, client: TestClient):
        """max_new_tokens = 0 returns 422."""
        response = client.post("/v1/generate", json={"prompt": "test", "max_new_tokens": 0})
        assert response.status_code == 422

    def test_generate_validation_temperature_out_of_range(self, client: TestClient):
        """Temperature > 2.0 returns 422."""
        response = client.post("/v1/generate", json={"prompt": "test", "temperature": 3.0})
        assert response.status_code == 422

    def test_generate_validation_negative_temperature(self, client: TestClient):
        """Negative temperature returns 422."""
        response = client.post("/v1/generate", json={"prompt": "test", "temperature": -0.5})
        assert response.status_code == 422

    def test_generate_validation_top_p_out_of_range(self, client: TestClient):
        """top_p > 1.0 returns 422."""
        response = client.post("/v1/generate", json={"prompt": "test", "top_p": 1.5})
        assert response.status_code == 422

    def test_generate_missing_prompt_field(self, client: TestClient):
        """Missing prompt field returns 422."""
        response = client.post("/v1/generate", json={"max_new_tokens": 10})
        assert response.status_code == 422

    def test_generate_long_prompt(self, client: TestClient):
        """Long prompt (within limit) works correctly."""
        long_prompt = "A " * 4000  # ~8000 chars, within 8192 limit
        response = client.post("/v1/generate", json={"prompt": long_prompt, "max_new_tokens": 5})
        assert response.status_code == 200
        assert "completion" in response.json()


class TestHealthEndpoint:
    """Tests for GET /v1/health."""

    def test_health_check_returns_healthy(self, client: TestClient):
        """Health check returns healthy status when model loaded."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "Qwen/Qwen3-0.6B"
        assert "version" in data

    def test_health_check_version_matches(self, client: TestClient):
        """Health check version matches app version."""
        response = client.get("/v1/health")
        assert response.json()["version"] == "1.0.0"


class TestModelInfoEndpoint:
    """Tests for GET /v1/model/info."""

    def test_model_info_returns_metadata(self, client: TestClient):
        """Model info returns expected metadata fields."""
        response = client.get("/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["loaded"] is True
        assert "dtype" in data
        assert "tensor_parallel_size" in data


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_basic_info(self, client: TestClient):
        """Root endpoint returns basic service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "atlas-foundry"
        assert data["version"] == "1.0.0"
        assert "author" in data
        assert data["author"] == "Alan Vo"