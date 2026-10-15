"""Консольный раннер агента — проверка без HTTP-сервиса.

    python -m app.agent.runner "What is Ridge regression?"

Поднимает LangGraph-агент в процессе (как /agent, но без FastAPI/Gradio),
прогоняет один вопрос и печатает ответ, список вызванных инструментов и счётчик
итераций. Удобно быстро проверить новый промпт/инструмент. ~10-20 сек на вопрос
(реальные LLM + Qdrant). Намеренно НЕ импортирует app.main, чтобы не тянуть gradio.
"""

import argparse
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent_graph
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.logging_config import get_logger

log = get_logger("task_api.agent.runner")


def run(question: str, thread_id: str = "cli") -> int:
    """Прогнать один вопрос через агента и напечатать результат. Возвращает exit-code."""
    try:
        check_input(question)
    except GuardrailError as e:
        print(f"⛔ Запрос отклонён guardrails: {e}", file=sys.stderr)
        return 2

    graph = build_agent_graph(checkpointer=MemorySaver())

    t0 = time.perf_counter()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "iteration_count": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    total_ms = int((time.perf_counter() - t0) * 1000)

    messages = result["messages"]
    tools_used = [
        tc["name"]
        for m in messages if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    raw_answer = messages[-1].content if messages else ""
    safe_answer, guardrail = check_output(str(raw_answer), tools_used)
    dedup_hits = sum(
        1 for m in messages
        if isinstance(m, ToolMessage) and "You already called" in str(m.content)
    )

    print("\n" + "=" * 70)
    print(f"❓ {question}")
    print("=" * 70)
    print(safe_answer)
    print("-" * 70)
    print(f"🛠  tools: {tools_used or '—'}")
    print(f"🔄 iterations: {result.get('iteration_count', 0)}")
    print(f"⏱  {total_ms} ms", end="")
    if dedup_hits:
        print(f" · dedup suppressed: {dedup_hits}", end="")
    if guardrail:
        print(f" · guardrail: {guardrail}", end="")
    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.agent.runner",
        description="Прогнать один вопрос через LangGraph-агента без HTTP.",
    )
    parser.add_argument("question", help="вопрос к агенту (в кавычках)")
    parser.add_argument(
        "--thread-id", default="cli", help="thread_id для памяти (по умолчанию 'cli')"
    )
    args = parser.parse_args()
    raise SystemExit(run(args.question, args.thread_id))


if __name__ == "__main__":
    main()
