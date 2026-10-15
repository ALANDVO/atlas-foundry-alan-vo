"""
Generation API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.inference import (
    GenerationRequest,
    GenerationResponse,
    InferenceService,
    get_inference_service,
)


router = APIRouter(prefix="/v1", tags=["generation"])


class GenerateRequestModel(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input prompt for generation")
    max_new_tokens: int = Field(default=32, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=0, ge=0, le=100, description="Top-k sampling (0 = disabled)")
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="Repetition penalty")
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v


class GenerateResponseModel(BaseModel):
    completion: str
    tokens_generated: int
    latency_ms: float
    model_name: str
    seed_used: int | None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_name: str


@router.post("/generate", response_model=GenerateResponseModel, status_code=status.HTTP_200_OK)
async def generate(
    request: GenerateRequestModel,
    service: InferenceService = Depends(get_inference_service),
) -> GenerateResponseModel:
    """
    Generate text completion for the given prompt.

    Returns deterministic output for same prompt + seed + parameters.
    """
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service initializing.",
        )

    gen_request = GenerationRequest(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
    )

    try:
        response = service.generate(gen_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )

    return GenerateResponseModel(
        completion=response.completion,
        tokens_generated=response.tokens_generated,
        latency_ms=response.latency_ms,
        model_name=response.model_name,
        seed_used=response.seed_used,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and model load state.
    """
    from app.core.config import settings

    return HealthResponse(
        status="healthy" if service.is_loaded else "initializing",
        version=settings.app_version,
        model_loaded=service.is_loaded,
        model_name=settings.model_name,
    )


@router.get("/model/info")
async def model_info(
    service: InferenceService = Depends(get_inference_service),
) -> dict:
    """
    Get model metadata.
    """
    return service.get_model_info()