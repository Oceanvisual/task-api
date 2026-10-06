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

    # Forward-прокси для запросов к LLM-провайдеру (Decodo и т.п.).
    # Нужен, когда IP сервиса режется geo-WAF провайдера (RU -> OpenRouter 403).
    # Формат: http://user:pass@host:port. Пусто = ходим напрямую.
    llm_proxy_url: str = ""

    # OpenRouter provider routing. DeepInfra для llama-3.3-70b на tool-calling
    # отдаёт finish_reason=error (пустой ответ, HTTP 200) — агент молча ломается.
    # Исключаем таких провайдеров; require_param_support роутит только на тех,
    # кто реально поддерживает переданные параметры (tools). Пусто = без правил.
    # OpenRouter-специфично; на других OpenAI-совместимых провайдерах оставить пустым.
    llm_ignore_providers: str = "DeepInfra"  # comma-separated
    llm_require_param_support: bool = True

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "sklearn_docs"
    top_k: int = 4

    # Embeddings (e5 — мультиязычный, нужно для русского)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    normalize_embeddings: bool = True

    # Логи (structlog). log_json=false -> человекочитаемый вывод в dev.
    log_level: str = "INFO"
    log_json: bool = True

    # agent
    max_iterations: int = 5
    agent_temperature: float = 0.0
    enable_web_search: bool = True
    # PythonREPLTool исполняет код в процессе сервиса = RCE через промпт
    # (os.environ, os.system). Включать только локально / в песочнице.
    enable_python_repl: bool = False
    agent_max_output_chars: int = 2000


settings = Settings()
