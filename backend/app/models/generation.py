"""
Generation request and response models.

These models define the contract for the text generation API.
"""

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


@dataclass(frozen=True)
class GenerationRequest:
    """
    Internal generation request - used by inference service.
    """
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenerationResponse:
    """
    Internal generation response - returned by inference service.
    """
    completion: str
    tokens_generated: int
    latency_ms: float
    model_name: str
    seed_used: Optional[int]


class GenerateRequestModel(BaseModel):
    """
    API request model for POST /v1/generate.
    """
    model_config = ConfigDict(protected_namespaces=())

    prompt: str = Field(..., min_length=1, max_length=8192, description="Input prompt for generation")
    max_new_tokens: int = Field(default=32, ge=1, le=2048, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=0, ge=0, le=100, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0, description="Repetition penalty")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducible generation")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v

    def to_internal(self) -> GenerationRequest:
        """Convert to internal GenerationRequest."""
        return GenerationRequest(
            prompt=self.prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )


class GenerateResponseModel(BaseModel):
    """
    API response model for POST /v1/generate.
    """
    model_config = ConfigDict(protected_namespaces=())

    completion: str = Field(..., description="Generated text completion")
    tokens_generated: int = Field(..., ge=0, description="Number of tokens generated")
    latency_ms: float = Field(..., ge=0.0, description="Generation latency in milliseconds")
    model_name: str = Field(..., description="Name of the model used")
    seed_used: Optional[int] = Field(default=None, description="Seed used for generation")

    @classmethod
    def from_internal(cls, response: GenerationResponse) -> "GenerateResponseModel":
        return cls(
            completion=response.completion,
            tokens_generated=response.tokens_generated,
            latency_ms=response.latency_ms,
            model_name=response.model_name,
            seed_used=response.seed_used,
        )
