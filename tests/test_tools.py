from unittest.mock import MagicMock, patch

from app.agent.tools import documentation_search, python_repl, web_search


@patch("app.agent.tools.build_rag_pipeline")
def test_documentation_search_returns_answer_and_sources(mock_build):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ridge is L2 regularization."
    doc = MagicMock(metadata={"source": "https://scikit-learn.org/ridge.html"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_build.return_value = (mock_chain, mock_retriever)

    # сбрасываем ленивый кэш, иначе подхватится состояние прошлого теста
    import app.agent.tools as tools_module

    tools_module._answer_chain = None
    tools_module._retriever = None

    result = documentation_search.invoke({"query": "What is Ridge?"})

    assert "Ridge" in result
    assert "Sources:" in result
    assert "scikit-learn.org" in result

    # single-retrieval: retrieval один раз, те же docs уходят в answer-цепочку
    mock_retriever.invoke.assert_called_once_with("What is Ridge?")
    mock_chain.invoke.assert_called_once_with(
        {"question": "What is Ridge?", "docs": [doc]}
    )


def test_python_repl_executes_arithmetic(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_python_repl", True)
    result = python_repl.invoke({"code": "print(7 * 6)"})
    assert "42" in result


def test_python_repl_handles_syntax_error_gracefully(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_python_repl", True)
    result = python_repl.invoke({"code": "print(1 +"})
    assert "error" in result.lower() or "syntax" in result.lower()


def test_web_search_respects_disable_flag(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", False)
    result = web_search.invoke({"query": "anything"})
    assert "disabled" in result.lower()


def test_python_repl_disabled_by_default_does_not_execute(monkeypatch):
    """REPL = RCE через промпт: по умолчанию выключен и код не исполняет."""
    import app.agent.tools as tools_module

    spy = MagicMock()
    monkeypatch.setattr(tools_module, "_python_repl", spy)
    result = python_repl.invoke({"code": "import os; print(os.environ)"})

    assert "disabled" in result.lower()
    spy.run.assert_not_called()
