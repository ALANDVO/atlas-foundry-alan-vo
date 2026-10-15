"""
Weight transfer request and response models.

These models define the contract for distributed weight synchronization
via NCCL for RL post-training loops.
"""

from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


@dataclass(frozen=True)
class WeightTransferInitRequest:
    """
    Internal request to initialize weight transfer engine.
    """
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str = "nccl"
    device_indices: Optional[List[int]] = None


@dataclass(frozen=True)
class WeightTransferInitResponse:
    """
    Internal response from weight transfer initialization.
    """
    success: bool
    message: str
    engine_rank: int
    world_size: int
    group_name: str


class WeightTransferInitRequestModel(BaseModel):
    """
    API request model for POST /init_weight_transfer_engine.

    Initializes the NCCL process group for weight updates.
    """
    master_address: str = Field(
        ...,
        description="TCP address of the master (trainer) process",
    )
    master_port: int = Field(
        ...,
        ge=1024,
        le=65535,
        description="TCP port for rendezvous",
    )
    rank_offset: int = Field(
        ...,
        ge=0,
        description="Rank offset for engine workers (trainer is rank 0)",
    )
    world_size: int = Field(
        ...,
        ge=2,
        description="Total world size (trainer + engine workers)",
    )
    group_name: str = Field(
        default="weight_update_group",
        description="NCCL process group name",
    )
    backend: str = Field(
        default="nccl",
        pattern="^(nccl|gloo)$",
        description="Distributed backend",
    )
    device_indices: Optional[List[int]] = Field(
        default=None,
        description="GPU device indices for each worker",
    )

    @field_validator("master_address")
    @classmethod
    def validate_master_address(cls, v: str) -> str:
        if not v:
            raise ValueError("master_address cannot be empty")
        return v

    @field_validator("group_name")
    @classmethod
    def validate_group_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("group_name cannot be empty")
        return v.strip()


class WeightTransferInitResponseModel(BaseModel):
    """
    API response model for POST /init_weight_transfer_engine.
    """
    success: bool
    message: str
    engine_rank: int
    world_size: int
    group_name: str


@dataclass(frozen=True)
class WeightUpdateStartRequest:
    """
    Internal request to start weight update session.
    """
    group_name: str


@dataclass(frozen=True)
class WeightUpdateStartResponse:
    """
    Internal response from starting weight update session.
    """
    success: bool
    message: str


class WeightUpdateStartRequestModel(BaseModel):
    """
    API request model for POST /start_weight_update.
    """
    group_name: str = Field(
        ...,
        description="NCCL process group name",
    )


class WeightUpdateStartResponseModel(BaseModel):
    """
    API response model for POST /start_weight_update.
    """
    success: bool
    message: str


@dataclass(frozen=True)
class WeightUpdateRequest:
    """
    Internal request for weight update.
    """
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str
    flush_cache: bool = False


@dataclass(frozen=True)
class WeightUpdateResponse:
    """
    Internal response from weight update.
    """
    success: bool
    message: str
    tensors_received: int
    bytes_received: int


class WeightUpdateRequestModel(BaseModel):
    """
    API request model for POST /update_weights.

    Receives broadcasted weight tensors via NCCL.
    """
    names: List[str] = Field(
        ...,
        min_length=1,
        description="Parameter names in order",
    )
    dtypes: List[str] = Field(
        ...,
        min_length=1,
        description="Data types (e.g., bfloat16, float32)",
    )
    shapes: List[List[int]] = Field(
        ...,
        min_length=1,
        description="Tensor shapes",
    )
    group_name: str = Field(
        ...,
        description="NCCL process group name",
    )
    flush_cache: bool = Field(
        default=False,
        description="Flush KV cache after weight update",
    )

    @field_validator("names", "dtypes", "shapes")
    @classmethod
    def validate_equal_length(cls, v: List, info) -> List:
        # Length validation happens at API level
        return v

    @field_validator("dtypes")
    @classmethod
    def validate_dtypes(cls, v: List[str]) -> List[str]:
        valid = {"float32", "float16", "bfloat16", "int8", "int32", "int64"}
        for dtype in v:
            if dtype not in valid:
                raise ValueError(f"Unsupported dtype: {dtype}. Valid: {valid}")
        return v


class WeightUpdateResponseModel(BaseModel):
    """
    API response model for POST /update_weights.
    """
    success: bool
    message: str
    tensors_received: int
    bytes_received: int


@dataclass(frozen=True)
class WeightUpdateFinishRequest:
    """
    Internal request to finish weight update session.
    """
    group_name: str


@dataclass(frozen=True)
class WeightUpdateFinishResponse:
    """
    Internal response from finishing weight update session.
    """
    success: bool
    message: str


class WeightUpdateFinishRequestModel(BaseModel):
    """
    API request model for POST /finish_weight_update.
    """
    group_name: str = Field(
        ...,
        description="NCCL process group name",
    )


class WeightUpdateFinishResponseModel(BaseModel):
    """
    API response model for POST /finish_weight_update.
    """
    success: bool
    message: str