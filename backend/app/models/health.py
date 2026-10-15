"""
Health and model info response models.
"""

from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(frozen=True)
class HealthResponse:
    """
    Internal health check response.
    """
    status: str
    version: str
    model_loaded: bool
    model_name: str


class HealthResponseModel(BaseModel):
    """
    API response model for GET /v1/health.
    """
    status: str
    version: str
    model_loaded: bool
    model_name: str

    @classmethod
    def from_internal(cls, response: HealthResponse) -> "HealthResponseModel":
        return cls(
            status=response.status,
            version=response.version,
            model_loaded=response.model_loaded,
            model_name=response.model_name,
        )


@dataclass(frozen=True)
class ModelInfoResponse:
    """
    Internal model info response.
    """
    model_name: str
    loaded: bool
    dtype: str
    tensor_parallel_size: int


class ModelInfoResponseModel(BaseModel):
    """
    API response model for GET /v1/model/info.
    """
    model_name: str
    loaded: bool
    dtype: str
    tensor_parallel_size: int

    @classmethod
    def from_internal(cls, response: ModelInfoResponse) -> "ModelInfoResponseModel":
        return cls(
            model_name=response.model_name,
            loaded=response.loaded,
            dtype=response.dtype,
            tensor_parallel_size=response.tensor_parallel_size,
        )