from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app, respond


def _mock_rag_pipeline(mock_build: MagicMock) -> None:
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "ok"
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_build.return_value = (mock_chain, mock_retriever)


def _mock_doc() -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"source": "https://scikit-learn.org/stable/linear.html"}
    doc.page_content = "Ridge regression addresses..."
    return doc


@patch("app.main.build_rag_pipeline")
def test_health(mock_build) -> None:
    _mock_rag_pipeline(mock_build)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.build_rag_pipeline")
def test_chat_validates_empty_question(mock_build) -> None:
    """Empty question must be rejected by Pydantic before reaching the chain."""
    _mock_rag_pipeline(mock_build)
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


@patch("app.main.build_rag_pipeline")
def test_chat_returns_answer_with_sources(mock_build) -> None:
    """Smoke test with fully mocked chain — no LLM call, no Qdrant call."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ridge uses L2 penalty [1]."

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [_mock_doc()]

    mock_build.return_value = (mock_chain, mock_retriever)

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is Ridge?"})

    assert response.status_code == 200
    body = response.json()
    assert "Ridge uses L2" in body["answer"]
    assert len(body["sources"]) == 1
    assert "scikit-learn.org" in body["sources"][0]["url"]


@patch("app.main.build_rag_pipeline")
def test_chat_retrieves_once_and_reuses_docs(mock_build) -> None:
    """Регрессия: один retrieval на запрос, те же docs уходят в LLM и в sources."""
    doc = _mock_doc()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "answer"
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_build.return_value = (mock_chain, mock_retriever)

    with TestClient(app) as client:
        client.post("/chat", json={"question": "What is Ridge?"})

    mock_retriever.invoke.assert_called_once_with("What is Ridge?")
    mock_chain.invoke.assert_called_once_with({"question": "What is Ridge?", "docs": [doc]})


@patch("app.main.build_rag_pipeline")
def test_respond_streams_with_single_retrieval(mock_build) -> None:
    """Gradio-стрим: retrieval один раз, стрим получает те же docs, ответ собирается из чанков."""
    doc = _mock_doc()
    mock_chain = MagicMock()
    mock_chain.stream.return_value = iter(["Ridge ", "uses L2."])
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_build.return_value = (mock_chain, mock_retriever)

    with TestClient(app):
        outputs = list(respond("What is Ridge?", []))

    mock_retriever.invoke.assert_called_once_with("What is Ridge?")
    mock_chain.stream.assert_called_once_with({"question": "What is Ridge?", "docs": [doc]})
    final_history = outputs[-1][0]
    assert final_history[-1] == {"role": "assistant", "content": "Ridge uses L2."}
