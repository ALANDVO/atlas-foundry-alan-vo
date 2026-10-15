"""
Inference Service - Core domain engine for text generation.

Round 1: Deterministic mock implementation for API contract validation.
Future: Real model loading with vLLM/TensorRT-LLM, tensor parallelism, weight updates.
"""

from dataclasses import dataclass
from typing import Optional
import time
import hashlib


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenerationResponse:
    completion: str
    tokens_generated: int
    latency_ms: float
    model_name: str
    seed_used: Optional[int]


class InferenceService:
    """
    Deterministic inference service for development and testing.

    Produces reproducible completions based on prompt hash and seed.
    No external LLM key required - this is the offline core.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self._loaded = False

    def load(self) -> None:
        """Load model (no-op in mock)."""
        self._loaded = True

    def unload(self) -> None:
        """Unload model."""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate completion for prompt.

        Deterministic: same prompt + seed + params = same output.
        This allows round-trip weight transfer testing.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.perf_counter()

        # Deterministic seed from request seed or prompt hash
        seed = request.seed
        if seed is None:
            seed = int(hashlib.md5(request.prompt.encode()).hexdigest()[:8], 16)

        # Mock completion - deterministic based on seed
        # In production, this calls the actual model
        completion = self._mock_generate(request.prompt, request.max_new_tokens, seed, request.temperature)

        latency_ms = (time.perf_counter() - start) * 1000
        tokens_generated = len(completion.split())

        return GenerationResponse(
            completion=completion,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            model_name=self.model_name,
            seed_used=seed,
        )

    def _mock_generate(self, prompt: str, max_tokens: int, seed: int, temperature: float) -> str:
        """
        Deterministic mock generation.

        Produces varied but reproducible output based on seed.
        Temperature=0 gives greedy (fixed) output; >0 adds variation.
        """
        # Simple deterministic "model" - cycles through word pool based on seed
        word_pool = [
            "Paris", "is", "the", "capital", "of", "France", "and", "a", "major",
            "European", "city", "known", "for", "art", "fashion", "gastronomy",
            "culture", "history", "architecture", "museums", "landmarks",
            "the", "Eiffel", "Tower", "Louvre", "Notre-Dame", "Seine", "river",
            "flows", "through", "heart", "city", "millions", "visit", "annually"
        ]

        # Deterministic shuffle based on seed
        import random
        rng = random.Random(seed)
        shuffled = word_pool.copy()
        rng.shuffle(shuffled)

        # Temperature effect: higher = more variation in selection
        if temperature == 0.0:
            # Greedy: just take first N words
            selected = shuffled[:max_tokens]
        else:
            # Sample with temperature-like variation
            selected = [rng.choice(shuffled) for _ in range(max_tokens)]

        return " ".join(selected)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "loaded": self._loaded,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
        }


# Global service instance (singleton pattern for FastAPI dependency injection)
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        from app.core.config import settings
        _inference_service = InferenceService(settings.model_name)
    return _inference_service


def set_inference_service(service: InferenceService) -> None:
    """For testing - replace the global service."""
    global _inference_service
    _inference_service = service