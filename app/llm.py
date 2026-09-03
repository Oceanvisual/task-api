"""Фабрика LLM-клиента.

Возвращает настроенный ChatOpenAI: все параметры (base_url, api_key,
модель, temperature) подтягиваются из `app.config.settings`. Работает с
любым OpenAI-совместимым провайдером — OpenRouter, Groq, Mistral, DeepSeek.
"""
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm() -> ChatOpenAI:
    """Собрать LLM-клиент с параметрами из settings."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
