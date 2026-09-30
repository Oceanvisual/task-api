"""Фабрика LLM-клиента.

Возвращает настроенный ChatOpenAI: все параметры (base_url, api_key,
модель, temperature) подтягиваются из `app.config.settings`. Работает с
любым OpenAI-совместимым провайдером — OpenRouter, Groq, Mistral, DeepSeek.

Если задан `llm_proxy_url`, весь трафик к провайдеру идёт через forward-прокси
(Decodo и т.п.) — обход geo-WAF, когда IP сервиса режется провайдером.
"""

from typing import Any

from langchain_openai import ChatOpenAI
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from app.config import settings


def get_llm() -> ChatOpenAI:
    """Собрать LLM-клиент с параметрами из settings."""
    kwargs: dict[str, Any] = {}
    if settings.llm_proxy_url:
        # Прокси-клиенты openai SDK (сохраняют дефолты SDK: limits, timeout,
        # follow_redirects — в отличие от голого httpx.Client). Передаём оба:
        # без http_async_client ChatOpenAI создаст async-клиент БЕЗ прокси, и
        # ainvoke/astream молча пошли бы напрямую -> снова 403 от geo-WAF.
        kwargs["http_client"] = DefaultHttpxClient(proxy=settings.llm_proxy_url)
        kwargs["http_async_client"] = DefaultAsyncHttpxClient(
            proxy=settings.llm_proxy_url
        )

    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        **kwargs,
    )
