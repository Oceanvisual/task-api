# 0005. OpenRouter + llama-3.3-70b с исключением провайдеров

- Дата: 2026-10-06
- Статус: Accepted
- Источник: app/config.py (комментарии к `llm_ignore_providers`), 30696e3, 25c2dc0

## Контекст
Агенту нужен tool-calling. DeepInfra для llama-3.3-70b на tool-calling отдаёт `finish_reason=error`
с пустым ответом и HTTP 200 — агент молча ломается.

## Решение
OpenAI-совместимый `ChatOpenAI` на OpenRouter, модель `meta-llama/llama-3.3-70b-instruct`, temperature 0.
`llm_ignore_providers="DeepInfra"`, `llm_require_param_support=True` (роутинг только на провайдеров с поддержкой tools).
Провайдер меняется только через Settings.

## Альтернативы
TODO(Misha): почему llama-3.3-70b, а не другая модель — не записано.

## Последствия
Новый провайдер с теми же симптомами → добавить в `llm_ignore_providers`, а не чинить в коде агента.
