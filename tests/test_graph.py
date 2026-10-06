from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_agent_graph


def _assert_no_dangling_tool_calls(messages):
    """Каждый tool_call закрыт ToolMessage — иначе OpenAI API вернёт 400 на след. ходе."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                assert tc["id"] in answered, f"dangling tool_call {tc['id']}"


def _looping_llm():
    counter = {"i": 0}

    def respond(*args, **kwargs):
        counter["i"] += 1
        return AIMessage(
            id=f"ai-{counter['i']}",
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": f"q{counter['i']}"},
                    "id": f"tc-{counter['i']}",
                }
            ],
        )

    llm = MagicMock()
    llm.invoke.side_effect = respond
    return llm


@patch("app.agent.graph._get_llm")
@patch("app.agent.graph._get_llm_with_tools")
def test_graph_terminates_after_max_iterations(mock_llm_factory, mock_plain_factory):
    """If LLM keeps returning tool_calls, graph must stop at max_iterations.

    NOTE: each AIMessage and tool_call must have a UNIQUE id. LangGraph's
    `add_messages` reducer deduplicates messages by id — если возвращать
    один и тот же объект из мока, second iteration «затирает» первое
    сообщение, last_msg оказывается ToolMessage, и should_continue
    выходит в END на iter=2.
    """
    counter = {"i": 0}

    def make_loop_response(*args, **kwargs):
        counter["i"] += 1
        return AIMessage(
            id=f"ai-{counter['i']}",
            content="",
            tool_calls=[
                {
                    "name": "python_repl",
                    "args": {"code": "print(1)"},
                    "id": f"tc-{counter['i']}",
                }
            ],
        )

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = make_loop_response
    mock_llm_factory.return_value = mock_llm
    mock_plain = MagicMock()
    mock_plain.invoke.return_value = AIMessage(content="Final from finalize")
    mock_plain_factory.return_value = mock_plain

    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="loop forever")], "iteration_count": 0},
        config={"configurable": {"thread_id": "test-loop"}, "recursion_limit": 50},
    )

    assert result["iteration_count"] >= 5
    # finalize: финальный текстовый ответ без tool_calls, не пустая заглушка
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == "Final from finalize"
    assert not last.tool_calls
    _assert_no_dangling_tool_calls(result["messages"])


def test_thread_stays_valid_after_max_iterations(monkeypatch):
    """Второй ход треда после упора в лимит получает валидную историю."""
    import app.agent.graph as graph_module

    monkeypatch.setattr("app.agent.graph.settings.max_iterations", 2)
    monkeypatch.setattr("app.agent.graph.settings.enable_web_search", False)
    looping = _looping_llm()
    monkeypatch.setattr(graph_module, "_get_llm_with_tools", lambda: looping)
    plain = MagicMock()
    plain.invoke.return_value = AIMessage(content="partial answer")
    monkeypatch.setattr(graph_module, "_get_llm", lambda: plain)

    graph = build_agent_graph()
    cfg = {"configurable": {"thread_id": "t-limit"}}
    graph.invoke({"messages": [HumanMessage(content="q1")], "iteration_count": 0}, cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="q2")], "iteration_count": 0}, cfg
    )

    _assert_no_dangling_tool_calls(result["messages"])
    # в finalize уходит история, где все tool_calls уже закрыты
    _assert_no_dangling_tool_calls(plain.invoke.call_args.args[0])


def test_finalize_falls_back_when_llm_fails(monkeypatch):
    import app.agent.graph as graph_module

    monkeypatch.setattr("app.agent.graph.settings.max_iterations", 1)
    monkeypatch.setattr("app.agent.graph.settings.enable_web_search", False)
    looping = _looping_llm()
    monkeypatch.setattr(graph_module, "_get_llm_with_tools", lambda: looping)
    plain = MagicMock()
    plain.invoke.side_effect = RuntimeError("provider down")
    monkeypatch.setattr(graph_module, "_get_llm", lambda: plain)

    result = build_agent_graph().invoke(
        {"messages": [HumanMessage(content="q")], "iteration_count": 0},
        {"configurable": {"thread_id": "t-fail"}},
    )

    assert "лимит шагов" in result["messages"][-1].content
    _assert_no_dangling_tool_calls(result["messages"])
