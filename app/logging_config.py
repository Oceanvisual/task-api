"""Конфигурация structlog для сервиса.

JSON в проде (`log_json=true`) — удобно для сбора в Loki/ELK; человекочитаемый
ConsoleRenderer в dev. Stdlib-логи (uvicorn, httpx) прогоняются через тот же
pipeline, чтобы формат был единым. Вызывать `configure_logging()` один раз на
старте приложения (в lifespan).
"""

import logging
import sys

import structlog

from app.config import settings

_configured = False


def configure_logging() -> None:
    """Идемпотентно настроить structlog + stdlib logging."""
    global _configured
    if _configured:
        return

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    # Процессоры (PrintLogger-совместимые, без stdlib-зависимостей вроде .name).
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Stdlib root -> stderr на том же уровне, чтобы uvicorn/httpx не терялись.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.getLevelName(settings.log_level.upper()),
    )

    _configured = True


def get_logger(name: str = "task_api"):
    """Получить bound-логгер с полем logger=<name>. Конфигурирует при первом обращении."""
    configure_logging()
    return structlog.get_logger().bind(logger=name)
