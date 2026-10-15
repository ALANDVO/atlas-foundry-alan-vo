"""
Atlas Foundry (TokenSpeed) - FastAPI Application.

A speed-of-light LLM inference engine with distributed weight updates
for RL post-training loops.

Author: Alan Vo | AI & Machine Learning
Contact: alanvo@gmail.com
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import generate, weight_transfer
from app.services.inference import get_inference_service, InferenceService
from app.services.weight_transfer import get_weight_transfer_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    # Startup
    inference_service = get_inference_service()
    inference_service.load()

    yield

    # Shutdown
    inference_service.unload()
    weight_transfer_service = get_weight_transfer_service()
    weight_transfer_service.shutdown()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory pattern allows consistent entrypoint for Docker and local commands.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="TokenSpeed - Speed-of-light LLM inference engine with distributed weight updates",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS for frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(generate.router)
    app.include_router(weight_transfer.router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "TokenSpeed - Speed-of-light LLM inference engine",
            "author": "Alan Vo",
            "docs": "/docs",
        }

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )