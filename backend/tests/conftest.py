"""
Backend test configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.services.inference import InferenceService, set_inference_service
from app.services.weight_transfer import WeightTransferService, set_weight_transfer_service


@pytest.fixture(scope="session")
def app():
    """Create FastAPI app for testing."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Test client for making requests."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def inference_service():
    """Provide a fresh inference service for each test."""
    service = InferenceService(model_name="test-model")
    service.load()
    set_inference_service(service)
    yield service
    service.unload()
    set_inference_service(None)


@pytest.fixture(autouse=True)
def weight_transfer_service():
    """Provide a fresh weight transfer service for each test."""
    service = WeightTransferService()
    set_weight_transfer_service(service)
    yield service
    service.shutdown()
    set_weight_transfer_service(None)