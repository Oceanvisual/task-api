# 0001. Зависимости в requirements.txt, не в uv-проекте

- Дата: 2026-07-15
- Статус: Accepted
- Источник: 965bc22 (chore: restructure for deployment, add requirements.txt); CLAUDE.md «Команды»

## Контекст
Docker и CI ставят зависимости через pip. torch нужен CPU-only через `--extra-index-url`
(иначе ~600 МБ CUDA-библиотек). uv с extra-index по умолчанию берёт пакеты (`requests` и др.)
только с индекса torch; sdist sqlalchemy 2.1.0 не собирается.

## Решение
`requirements.txt` — единственный источник правды; `pyproject.toml` без deps (только конфиг инструментов).
Локально uv используется как быстрый pip: `uv pip install --index-strategy unsafe-best-match --only-binary sqlalchemy -r requirements.txt`.

## Альтернативы
- uv-проект с `uv.lock`: отложено, требует настройки индексов torch в `[tool.uv]` и проверки Docker-сборки.

## Последствия
Глобальное правило «только uv add» здесь не действует. `uv.lock` в рабочем дереве не используется.
Пересмотреть: при миграции Docker/CI на `uv sync --locked`.
