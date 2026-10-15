from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Вопрос к агенту. thread_id связывает ходы одного диалога (checkpointer)."""

    question: str = Field(..., min_length=1, max_length=500)
    thread_id: str | None = Field(
        default=None,
        description="ID диалога для памяти агента; если не задан — генерируется.",
    )


class TraceStep(BaseModel):
    """Один шаг ReAct-трейса: какой тул вызвал агент и что получил."""

    step: int
    node: str
    tool: str
    input: dict[str, Any]
    output: str
    latency_ms: int


class Source(BaseModel):
    """Источник, пришедший из documentation_search (URL + контекст чанка)."""

    url: str
    snippet: str = Field(..., description="Первые ~200 символов контекста")
    full_context: str = Field(..., description="Полный текст, который получил агент")


class AgentResponse(BaseModel):
    """Финальный ответ агента с трейсом, источниками и статусом guardrails."""

    answer: str
    trace: list[TraceStep]
    sources: list[Source]
    guardrail_triggered: str | None = None
    iterations: int = 0
