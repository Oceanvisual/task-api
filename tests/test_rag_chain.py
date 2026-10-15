import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

from app.rag.chain import build_answer_chain, build_rag_chain

DOCS = [
    Document(page_content="Ridge adds an L2 penalty.", metadata={"source": "linear.html"}),
    Document(page_content="Lasso adds an L1 penalty.", metadata={"source": "lasso.html"}),
]


def _capturing_llm(prompts: list[str]) -> RunnableLambda:
    """Fake LLM: запоминает отрендеренный prompt и возвращает фиксированный ответ."""

    def _call(prompt: PromptValue) -> AIMessage:
        prompts.append(prompt.to_string())
        return AIMessage("Ridge uses L2 [1].")

    return RunnableLambda(_call)


def test_answer_chain_uses_given_docs_as_context() -> None:
    prompts: list[str] = []
    chain = build_answer_chain(_capturing_llm(prompts))

    answer = chain.invoke({"question": "What is Ridge?", "docs": DOCS})

    assert answer == "Ridge uses L2 [1]."
    assert "[1] Source: linear.html\nRidge adds an L2 penalty." in prompts[0]
    assert "[2] Source: lasso.html" in prompts[0]
    assert "Question: What is Ridge?" in prompts[0]


def test_rag_chain_keeps_str_interface_for_eval_notebook() -> None:
    """notebooks/rag_eval.ipynb вызывает chain.invoke(question): интерфейс str -> str сохранён."""
    prompts: list[str] = []
    retriever = MagicMock()
    retriever.invoke.return_value = DOCS
    vectorstore = MagicMock()
    vectorstore.as_retriever.return_value = RunnableLambda(retriever.invoke)

    with (
        patch("app.rag.chain.get_vectorstore", return_value=vectorstore),
        patch("app.rag.chain.get_llm", return_value=_capturing_llm(prompts)),
    ):
        chain, _ = build_rag_chain()
        answer = chain.invoke("What is Ridge?")

    assert answer == "Ridge uses L2 [1]."
    retriever.invoke.assert_called_once()
    assert "Ridge adds an L2 penalty." in prompts[0]


def test_gradio_ui_has_no_ignored_blocks_params() -> None:
    """Gradio 6 молча игнорирует theme/css в gr.Blocks(...): импорт app.main не должен это триггерить."""
    env = {**os.environ, "LLM_API_KEY": "test-key-not-used"}
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error:The parameters have been moved:UserWarning",
            "-c",
            "import app.main",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
