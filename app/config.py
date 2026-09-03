from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Только секрет — без default, контейнер падает если не задан.
    llm_api_key: str

    # LLM provider. Дефолты для OpenRouter; override через .env при смене провайдера.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    llm_temperature: float = 0.0

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "sklearn_docs"
    top_k: int = 4

    # Embeddings (e5 — мультиязычный, нужно для русского)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    normalize_embeddings: bool = True


settings = Settings()
