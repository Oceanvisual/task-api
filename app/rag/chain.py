"""LCEL-цепочка RAG: retriever → prompt → LLM → parser.

`build_rag_chain()` собирает цепочку из готовых блоков и возвращает её
вместе с retriever'ом (он отдельно нужен для оценки в Шаге 8).
`format_docs_with_sources()` склеивает топ-k чанков в нумерованный
контекст, чтобы LLM могла цитировать источники как `[1]`, `[2]`.
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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


def build_rag_chain():
    """Собрать LCEL-цепочку и вернуть пару (chain, retriever)."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs_with_sources),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
