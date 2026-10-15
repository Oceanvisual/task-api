"""LCEL-цепочка RAG: retriever → prompt → LLM → parser.

`build_rag_pipeline()` возвращает answer-цепочку (принимает уже найденные
документы) и retriever: сервис делает retrieval один раз и переиспользует
документы и для ответа, и для панели источников.
`build_rag_chain()` — полная цепочка `question -> answer` для ноутбука оценки.
`format_docs_with_sources()` склеивает топ-k чанков в нумерованный
контекст, чтобы LLM могла цитировать источники как `[1]`, `[2]`.
"""

from operator import itemgetter
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings
from app.llm import get_llm

SYSTEM_PROMPT = """You are a study assistant for the Classic ML cycle of an ML/DS course.
The context below is taken from the official scikit-learn documentation and from
the service's internal "about" pages.

Rules:
- Use ONLY the provided context. If the answer is not in the context, say so honestly.
- Cite sources using [1], [2], ... — the numbers correspond to the source list in the context block.
- Reply in the SAME LANGUAGE as the user's question (English question -> English answer,
  Russian question -> Russian answer). Translate the relevant facts; keep code identifiers
  (function names, parameter names, classes) in English.
- If the user asks meta-questions ("what do you know about?", "что ты умеешь?") —
  answer based on the internal "About this RAG assistant" context.

Context:
{context}

Question: {question}

Answer (with citations):"""


def get_vectorstore() -> QdrantVectorStore:
    """Поднять клиент Qdrant + эмбеддер и завернуть в LangChain-VectorStore."""
    client = QdrantClient(url=settings.qdrant_url)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": settings.normalize_embeddings},
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )


def format_docs_with_sources(docs: list[Document]) -> str:
    """Склеить топ-k чанков в нумерованный context-блок для prompt'а LLM."""
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(lines)


class AnswerInput(TypedDict):
    """Вход answer-цепочки: вопрос и уже найденные retriever'ом документы."""

    question: str
    docs: list[Document]


def build_answer_chain(llm: LanguageModelLike) -> Runnable[AnswerInput, str]:
    """prompt → LLM → parser поверх готовых документов (без retrieval внутри)."""
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    return (
        {
            "context": itemgetter("docs") | RunnableLambda(format_docs_with_sources),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_pipeline() -> tuple[Runnable[AnswerInput, str], BaseRetriever]:
    """Вернуть (answer_chain, retriever): retrieval делает вызывающий код, один раз."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})
    return build_answer_chain(get_llm()), retriever


def build_rag_chain() -> tuple[Runnable[str, str], BaseRetriever]:
    """Полная цепочка `question -> answer` + retriever (для notebooks/rag_eval.ipynb)."""
    answer_chain, retriever = build_rag_pipeline()
    chain = {"question": RunnablePassthrough(), "docs": retriever} | answer_chain
    return chain, retriever
