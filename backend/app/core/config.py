from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # Application
    app_name: str = "atlas-foundry"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    # Model settings (for future phases)
    model_name: str = "Qwen/Qwen3-0.6B"
    model_dtype: str = "bfloat16"
    tensor_parallel_size: int = 1

    # LLM Provider settings (opt-in, for advisory explanations)
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_provider: str = "openai-compatible"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    # Distributed weight transfer
    nccl_master_address: str = "127.0.0.1"
    nccl_master_port: int = 29511
    weight_update_group_name: str = "weight_update_group"

    # Auth (future phase)
    oidc_issuer_url: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None

    # Demo mode (local-only, refuses in production)
    demo_mode: bool = False


settings = Settings()
