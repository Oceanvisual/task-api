from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.agent.graph as graph_module
from app.agent.graph import _tool_sig, build_agent_graph, tool_executor_node


def _ai_with_tool(name, args, call_id, msg_id):
    return AIMessage(
        id=msg_id,
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def test_tool_executor_runs_fresh_call(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_python_repl", True)
    state = {
        "messages": [_ai_with_tool("python_repl", {"code": "print(2+2)"}, "c1", "a1")],
        "iteration_count": 1,
        "seen_tool_sigs": [],
    }
    out = tool_executor_node(state)
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "4" in msg.content
    assert out["seen_tool_sigs"] == [_tool_sig("python_repl", {"code": "print(2+2)"})]


def test_tool_executor_dedupes_repeated_call():
    sig = _tool_sig("python_repl", {"code": "print(2+2)"})
    state = {
        "messages": [_ai_with_tool("python_repl", {"code": "print(2+2)"}, "c2", "a2")],
        "iteration_count": 2,
        "seen_tool_sigs": [sig],  # уже исполнялся
    }
    out = tool_executor_node(state)
    msg = out["messages"][0]
    assert "already" in msg.content.lower()
    assert "4" not in msg.content  # тул НЕ исполнялся
    assert out["seen_tool_sigs"] == []  # дубль не добавляет новую сигнатуру


class _LoopLLM:
    """Всегда просит один и тот же documentation_search — имитация зацикливания."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _ai_with_tool(
            "documentation_search",
            {"query": "Ridge"},
            f"c{self.calls}",
            f"a{self.calls}",
        )


def test_dedup_breaks_loop_tool_invoked_once(monkeypatch):
    """LLM бесконечно дёргает тот же тул, но реальный тул зовётся РОВНО раз."""
    monkeypatch.setattr("app.agent.graph.settings.max_iterations", 5)

    spy = MagicMock()
    spy.invoke.return_value = "Ridge is L2 regularization."
    monkeypatch.setattr(graph_module, "_TOOLS_BY_NAME", {"documentation_search": spy})

    fake = _LoopLLM()
    monkeypatch.setattr(graph_module, "_get_llm_with_tools", lambda: fake)

    app = build_agent_graph()
    result = app.invoke(
        {"messages": [HumanMessage(content="Ridge?")], "iteration_count": 0},
        config={"configurable": {"thread_id": "dedup-1"}},
    )

    # реальный тул исполнен один раз, дальше — дедуп-заглушки
    spy.invoke.assert_called_once()
    assert (
        result["iteration_count"] == 5
    )  # дошли до backstop, но без лишних вызовов тула
    dup_msgs = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and "already" in m.content.lower()
    ]
    assert dup_msgs  # были дедуп-ответы
