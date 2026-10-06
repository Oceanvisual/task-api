import json
import operator
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.config import settings
from app.llm import get_llm
from app.logging_config import get_logger

log = get_logger("task_api.agent.graph")

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    # сигнатуры уже исполненных (tool, args) — против зацикливания; operator.add аккумулирует
    seen_tool_sigs: Annotated[list[str], operator.add]


_llm = None
_llm_with_tools = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(TOOLS)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _get_llm_with_tools().invoke(messages)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    iteration = state["iteration_count"] + 1
    tool_calls = getattr(response, "tool_calls", None) or []
    finish_reason = (getattr(response, "response_metadata", {}) or {}).get(
        "finish_reason"
    )
    log.info(
        "agent_step",
        iteration=iteration,
        latency_ms=latency_ms,
        tool_calls_count=len(tool_calls),
        requested_tools=[tc["name"] for tc in tool_calls],
        finish_reason=finish_reason,
        has_content=bool(response.content),
    )
    # finish_reason=error от провайдера -> пустой ответ, иначе теряется молча
    if finish_reason == "error" and not tool_calls and not response.content:
        log.error("llm_finish_error", iteration=iteration)
    return {
        "messages": [response],
        "iteration_count": iteration,
    }


def _tool_sig(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def tool_executor_node(state: AgentState) -> dict:
    """Исполняет tool_calls последнего AIMessage с дедупликацией.

    Если агент повторно запрашивает тот же (tool, args), что уже исполнялся в
    этом треде, НЕ зовём тул снова, а возвращаем ToolMessage-инструкцию:
    используй прошлый результат и отвечай. Это рубит самый частый цикл, когда
    LLM дёргает documentation_search с одним запросом до max_iterations.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    seen = set(state.get("seen_tool_sigs", []))

    out_messages: list[ToolMessage] = []
    new_sigs: list[str] = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        sig = _tool_sig(name, args)

        if sig in seen:
            log.warning("duplicate_tool_call", tool=name, args=args)
            content = (
                f"⚠️ You already called `{name}` with identical arguments earlier. "
                "Do NOT call it again. Use the previous result above and answer the "
                "user now; if it's not sufficient, answer from your own knowledge and "
                "say so explicitly."
            )
        else:
            tool = _TOOLS_BY_NAME.get(name)
            if tool is None:
                log.error("unknown_tool", tool=name)
                content = f"Error: unknown tool {name}"
            else:
                try:
                    content = str(tool.invoke(args))
                except Exception as exc:  # noqa: BLE001 — ошибку тула возвращаем агенту
                    log.error("tool_error", tool=name, exc_type=type(exc).__name__)
                    content = f"Error executing {name}: {exc}"
            new_sigs.append(sig)
            seen.add(sig)

        out_messages.append(
            ToolMessage(content=content, tool_call_id=tc["id"], name=name)
        )

    return {"messages": out_messages, "seen_tool_sigs": new_sigs}


_STEP_LIMIT_NOTE = (
    "Step limit reached: this tool call was NOT executed. Answer the user now "
    "using the results above; say explicitly if something is missing."
)
_STEP_LIMIT_FALLBACK = "Не уложился в лимит шагов агента. Переформулируй вопрос точнее или разбей его на части."


def finalize_node(state: AgentState) -> dict:
    """Лимит итераций при висящих tool_calls: закрыть их и ответить без тулов.

    Если просто выйти в END, в checkpointer остаётся AIMessage с tool_calls без
    ответных ToolMessage, и следующий ход этого треда OpenAI-совместимый API
    отклоняет с 400. Поэтому закрываем каждый tool_call заглушкой и даём LLM
    (без bind_tools) сформулировать ответ по уже собранным результатам.
    """
    last = state["messages"][-1]
    stubs = [
        ToolMessage(content=_STEP_LIMIT_NOTE, tool_call_id=tc["id"], name=tc["name"])
        for tc in last.tool_calls
    ]
    log.warning("max_iterations_reached", iteration=state["iteration_count"])
    try:
        response = _get_llm().invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"], *stubs]
        )
    except Exception as exc:  # noqa: BLE001 — тред должен остаться валидным
        log.error("finalize_llm_error", exc_type=type(exc).__name__)
        response = AIMessage(content=_STEP_LIMIT_FALLBACK)
    if not response.content:
        response = AIMessage(content=_STEP_LIMIT_FALLBACK)
    return {"messages": [*stubs, response]}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if not getattr(last_msg, "tool_calls", None):
        return END
    if state["iteration_count"] >= settings.max_iterations:
        return "finalize"
    return "tool_executor"


def build_agent_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
