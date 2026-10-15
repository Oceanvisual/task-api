import structlog
from structlog.testing import capture_logs

from app.logging_config import configure_logging, get_logger


def test_configure_is_idempotent():
    configure_logging()
    configure_logging()  # второй вызов не падает/не переинициализирует
    assert structlog.is_configured()


def test_logger_emits_event_with_fields():
    log = get_logger("task_api.test")
    with capture_logs() as logs:
        log.info("unit_event", foo="bar", n=1)
    assert len(logs) == 1
    entry = logs[0]
    assert entry["event"] == "unit_event"
    assert entry["foo"] == "bar"
    assert entry["n"] == 1
    assert entry["log_level"] == "info"
    assert entry["logger"] == "task_api.test"  # имя прибандено в get_logger


def test_contextvars_bind_and_clear():
    structlog.contextvars.bind_contextvars(thread_id="t-42")
    try:
        assert structlog.contextvars.get_contextvars()["thread_id"] == "t-42"
    finally:
        structlog.contextvars.clear_contextvars()
    assert "thread_id" not in structlog.contextvars.get_contextvars()
