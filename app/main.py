import time
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.rag.chain import build_rag_pipeline
from app.schemas.chat import ChatRequest, ChatResponse, Source

# answer-цепочка принимает {"question", "docs"}: retrieval делаем один раз
# и отдаём те же документы и в LLM, и в панель источников.
_answer_chain = None
_retriever = None

# LaTeX delimiters для Gradio Chatbot. LLM-ответы про Ridge, Lasso, метрики
# содержат формулы $$..$$ / \[..\] / $..$ — без этого блока они отрисуются
# как сырые строки `$\ell_1$`.
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _answer_chain, _retriever
    _answer_chain, _retriever = build_rag_pipeline()
    print("RAG chain ready")
    yield
    _answer_chain = None
    _retriever = None


def _provider_error(exc: Exception) -> tuple[int | None, str]:
    """Достаёт HTTP-статус и текст ошибки провайдера из исключения OpenAI SDK.

    `openai.APIStatusError` (и подклассы вроде PermissionDeniedError) несут
    `.status_code`, распарсенный `.body` и `.response`. Раньше логировался
    только тип исключения — тело с реальной причиной (data policy / no credits /
    model not allowed) терялось. Возвращает (status, message) для логов и ответа.
    """
    status = getattr(exc, "status_code", None)
    message: object = None

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            message = err.get("message") or err.get("code")
        elif isinstance(err, str):
            message = err

    if not message:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                message = response.text
            except Exception:  # noqa: BLE001 — best-effort извлечение тела
                message = None

    if not message:
        message = getattr(exc, "message", None) or str(exc)

    # str() — message из body может быть не-строкой (int-code, dict);
    # схлопываем переносы (log injection / поломка markdown code-fence)
    # и обрезаем — тело может быть HTML от WAF на сотни КБ.
    clean = " ".join(str(message).split())[:300]
    return status, clean


app = FastAPI(title="RAG service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    docs = _retriever.invoke(payload.question)
    try:
        answer = _answer_chain.invoke({"question": payload.question, "docs": docs})
    except Exception as exc:
        status, message = _provider_error(exc)
        print(
            f"LLM error in /chat: {type(exc).__name__} status={status} detail={message}"
        )
        detail = (
            "LLM provider temporarily unavailable. Try again in 30-60 seconds. "
            f"Raw: {type(exc).__name__}"
        )
        if status:
            detail += f" ({status})"
        if message:
            detail += f": {message}"
        raise HTTPException(status_code=503, detail=detail) from exc
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


def _format_timings(
    retrieval_ms: float, llm_ms: float | None, llm_error: str | None
) -> str:
    lines = [
        "### ⏱ Тайминги последнего запроса",
        "",
        f"- 🔍 **Retrieval (embed + Qdrant):** {retrieval_ms:.0f} ms",
    ]
    if llm_ms is not None:
        lines.append(f"- 🤖 **LLM call:** {llm_ms:.0f} ms")
        lines.append(f"- 📊 **Total:** {retrieval_ms + llm_ms:.0f} ms")
    else:
        lines.append(f"- 🤖 **LLM call:** ❌ {llm_error}")
    return "\n".join(lines)


def _format_sources(docs: list) -> str:
    if not docs:
        return "### 📚 Источники\n\n_Ничего не найдено_"
    lines = ["### 📚 Источники", ""]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:140].strip().replace("\n", " ")
        lines.append(f"**[{i}]** `{source}`")
        lines.append(f"> {snippet}…")
        lines.append("")
    return "\n".join(lines)


def respond(message: str, history: list):
    """Streaming Gradio handler — generator that yields on every chunk."""
    if not message or not message.strip():
        yield (
            history,
            "",
            "### ⏱ Тайминги\n\n_Пустой запрос_",
            "### 📚 Источники\n\n_—_",
        )
        return

    history = history + [{"role": "user", "content": message}]

    t0 = time.perf_counter()
    docs = _retriever.invoke(message)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    sources_panel = _format_sources(docs)

    # Yield #1: sources уже на экране, LLM ещё не начал писать.
    history.append({"role": "assistant", "content": ""})
    yield (
        history,
        "",
        (
            "### ⏱ Тайминги\n\n"
            f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
            "- 🤖 **LLM:** _streaming…_"
        ),
        sources_panel,
    )

    t1 = time.perf_counter()
    ttft_ms: float | None = None
    accumulated = ""
    try:
        for chunk in _answer_chain.stream({"question": message, "docs": docs}):
            if not chunk:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t1) * 1000
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield (
                history,
                "",
                (
                    f"### ⏱ Тайминги\n\n"
                    f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
                    f"- ⚡ **TTFT (1st token):** {ttft_ms:.0f} ms\n"
                    f"- 🤖 **LLM:** _streaming… {len(accumulated)} chars_"
                ),
                sources_panel,
            )

        llm_total_ms = (time.perf_counter() - t1) * 1000
        yield (
            history,
            "",
            (
                "### ⏱ Тайминги последнего запроса\n\n"
                f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
                f"- ⚡ **TTFT:** {ttft_ms:.0f} ms\n"
                f"- 🤖 **LLM stream (full):** {llm_total_ms:.0f} ms\n"
                f"- 📊 **Total:** {retrieval_ms + llm_total_ms:.0f} ms"
            ),
            sources_panel,
        )
    except Exception as exc:  # noqa: BLE001 — любую ошибку провайдера показываем в UI
        status, message = _provider_error(exc)
        print(
            f"LLM error in respond: {type(exc).__name__} "
            f"status={status} detail={message}"
        )
        detail = type(exc).__name__
        if status:
            detail += f" {status}"
        content = (
            f"⚠️ LLM-провайдер сейчас недоступен ({detail}). "
            "Попробуй через 30-60 секунд."
        )
        if message:
            content += f"\n\n```\n{message}\n```"
        history[-1]["content"] = content
        yield (
            history,
            "",
            _format_timings(retrieval_ms, None, detail),
            sources_panel,
        )


# CSS делает три вещи: 1) распахивает контейнер на всю ширину,
# 2) фиксирует высоту чата и боковой панели на calc(100vh - 220px) —
#    минус headers — чтобы при наборе сообщения чат НЕ сжимался,
# 3) добавляет видимую границу между чатом и боковой панелью.
CSS = """
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
#chatbot { height: calc(100vh - 220px) !important; min-height: 500px !important; }
#side-panel { height: calc(100vh - 220px) !important; overflow-y: auto !important;
              padding: 1rem !important; border-left: 1px solid #ddd !important; }
"""

# Gradio 6: theme/css задаются при запуске/монтировании, а не в gr.Blocks(...)
# (иначе молча игнорируются). См. mount_gradio_app ниже.
with gr.Blocks(
    title="scikit-learn docs RAG",
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 📖 scikit-learn docs RAG assistant\n"
        "_Спрашивай про Linear models, Decision trees, Metrics — на русском или английском._"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                latex_delimiters=LATEX_DELIMITERS,
                buttons=["copy"],
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                    scale=8,
                    container=False,
                    autofocus=True,
                )
                send = gr.Button("Отправить", scale=1, variant="primary")
            gr.Examples(
                examples=[
                    "How does Ridge regression work?",
                    "Что ты умеешь?",
                    "Объясни разницу между precision и recall с формулами",
                    "When does a decision tree overfit?",
                ],
                inputs=msg,
            )
        with gr.Column(scale=1, elem_id="side-panel"):
            timings_md = gr.Markdown(
                "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
            )
            sources_md = gr.Markdown("### 📚 Источники\n\n_—_")

    msg.submit(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])
    send.click(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])


demo.queue()
app = gr.mount_gradio_app(app, demo, path="/ui", css=CSS, theme=gr.themes.Soft())
