"""
Tests for the weight transfer API endpoints.

Covers:
- Initialize weight transfer engine
- Start weight update session
- Update weights (NCCL broadcast receive)
- Finish weight update session
- Validation errors
- Session lifecycle
"""

import pytest
from fastapi.testclient import TestClient


class TestWeightTransferInit:
    """Tests for POST /v1/init_weight_transfer_engine."""

    def test_init_success(self, client: TestClient):
        """Successful initialization returns engine config."""
        response = client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_update_group",
                "backend": "nccl",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["engine_rank"] == 1
        assert data["world_size"] == 2
        assert data["group_name"] == "weight_update_group"

    def test_init_invalid_backend(self, client: TestClient):
        """Invalid backend returns 422."""
        response = client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_update_group",
                "backend": "invalid",
            },
        )
        assert response.status_code == 422

    def test_init_missing_master_address(self, client: TestClient):
        """Missing master_address returns 422."""
        response = client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_update_group",
            },
        )
        assert response.status_code == 422

    def test_init_world_size_too_small(self, client: TestClient):
        """world_size < 2 returns 422."""
        response = client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 1,
                "group_name": "weight_update_group",
            },
        )
        assert response.status_code == 422

    def test_init_empty_group_name(self, client: TestClient):
        """Empty group_name returns 422."""
        response = client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "",
            },
        )
        assert response.status_code == 422


class TestWeightTransferStart:
    """Tests for POST /v1/start_weight_update."""

    def test_start_success(self, client: TestClient):
        """Start weight update after init succeeds."""
        # First initialize
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )

        # Then start update
        response = client.post(
            "/v1/start_weight_update",
            json={"group_name": "test_group"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "started" in data["message"].lower()

    def test_start_without_init(self, client: TestClient):
        """Start without init returns 503."""
        response = client.post(
            "/v1/start_weight_update",
            json={"group_name": "nonexistent_group"},
        )
        assert response.status_code == 503

    def test_start_group_mismatch(self, client: TestClient):
        """Start with different group name returns 400."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "group_a",
            },
        )

        response = client.post(
            "/v1/start_weight_update",
            json={"group_name": "group_b"},
        )
        assert response.status_code == 400


class TestWeightTransferUpdate:
    """Tests for POST /v1/update_weights."""

    def test_update_success(self, client: TestClient):
        """Update weights after init and start succeeds."""
        # Initialize
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )

        # Start
        client.post("/v1/start_weight_update", json={"group_name": "test_group"})

        # Update
        response = client.post(
            "/v1/update_weights",
            json={
                "names": ["model.embed_tokens.weight"],
                "dtypes": ["bfloat16"],
                "shapes": [[151936, 1024]],
                "group_name": "test_group",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tensors_received"] == 1
        assert data["bytes_received"] > 0

    def test_update_without_start(self, client: TestClient):
        """Update without start returns 503."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )

        response = client.post(
            "/v1/update_weights",
            json={
                "names": ["param1"],
                "dtypes": ["float32"],
                "shapes": [[10, 10]],
                "group_name": "test_group",
            },
        )
        assert response.status_code == 503

    def test_update_mismatched_lengths(self, client: TestClient):
        """Mismatched names/dtypes/shapes lengths returns 400."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )
        client.post("/v1/start_weight_update", json={"group_name": "test_group"})

        response = client.post(
            "/v1/update_weights",
            json={
                "names": ["param1", "param2"],
                "dtypes": ["float32"],
                "shapes": [[10, 10]],
                "group_name": "test_group",
            },
        )
        assert response.status_code == 400

    def test_update_invalid_dtype(self, client: TestClient):
        """Invalid dtype returns 422."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )
        client.post("/v1/start_weight_update", json={"group_name": "test_group"})

        response = client.post(
            "/v1/update_weights",
            json={
                "names": ["param1"],
                "dtypes": ["invalid_dtype"],
                "shapes": [[10, 10]],
                "group_name": "test_group",
            },
        )
        assert response.status_code == 422

    def test_update_empty_names(self, client: TestClient):
        """Empty names list returns 422."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )
        client.post("/v1/start_weight_update", json={"group_name": "test_group"})

        response = client.post(
            "/v1/update_weights",
            json={
                "names": [],
                "dtypes": [],
                "shapes": [],
                "group_name": "test_group",
            },
        )
        assert response.status_code == 422


class TestWeightTransferFinish:
    """Tests for POST /v1/finish_weight_update."""

    def test_finish_success(self, client: TestClient):
        """Finish weight update session."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )
        client.post("/v1/start_weight_update", json={"group_name": "test_group"})
        client.post(
            "/v1/update_weights",
            json={
                "names": ["param1"],
                "dtypes": ["float32"],
                "shapes": [[10, 10]],
                "group_name": "test_group",
            },
        )

        response = client.post(
            "/v1/finish_weight_update",
            json={"group_name": "test_group"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "finished" in data["message"].lower()

    def test_finish_without_start(self, client: TestClient):
        """Finish without start returns 503."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "test_group",
            },
        )

        response = client.post(
            "/v1/finish_weight_update",
            json={"group_name": "test_group"},
        )
        assert response.status_code == 503

    def test_finish_group_mismatch(self, client: TestClient):
        """Finish with different group name returns 400."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "group_a",
            },
        )
        client.post("/v1/start_weight_update", json={"group_name": "group_a"})

        response = client.post(
            "/v1/finish_weight_update",
            json={"group_name": "group_b"},
        )
        assert response.status_code == 400


class TestWeightTransferLifecycle:
    """Tests for complete weight transfer lifecycle."""

    def test_full_lifecycle(self, client: TestClient):
        """Complete init -> start -> update -> finish lifecycle."""
        # Init
        response = client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 3,
                "group_name": "lifecycle_test",
                "backend": "nccl",
                "device_indices": [0, 1],
            },
        )
        assert response.status_code == 200
        assert response.json()["engine_rank"] == 1

        # Start
        response = client.post("/v1/start_weight_update", json={"group_name": "lifecycle_test"})
        assert response.status_code == 200

        # Update (multiple tensors)
        response = client.post(
            "/v1/update_weights",
            json={
                "names": [
                    "model.embed_tokens.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                ],
                "dtypes": ["bfloat16", "bfloat16", "bfloat16"],
                "shapes": [
                    [151936, 1024],
                    [1024, 1024],
                    [1024, 1024],
                ],
                "group_name": "lifecycle_test",
                "flush_cache": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tensors_received"] == 3
        assert data["bytes_received"] > 0

        # Finish
        response = client.post("/v1/finish_weight_update", json={"group_name": "lifecycle_test"})
        assert response.status_code == 200

    def test_multiple_update_calls(self, client: TestClient):
        """Multiple update_weights calls in one session."""
        client.post(
            "/v1/init_weight_transfer_engine",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29511,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "multi_update",
            },
        )
        client.post("/v1/start_weight_update", json={"group_name": "multi_update"})

        # First update
        response = client.post(
            "/v1/update_weights",
            json={
                "names": ["param1"],
                "dtypes": ["float32"],
                "shapes": [[10, 10]],
                "group_name": "multi_update",
            },
        )
        assert response.status_code == 200

        # Second update
        response = client.post(
            "/v1/update_weights",
            json={
                "names": ["param2"],
                "dtypes": ["float32"],
                "shapes": [[20, 20]],
                "group_name": "multi_update",
            },
        )
        assert response.status_code == 200

        client.post("/v1/finish_weight_update", json={"group_name": "multi_update"})