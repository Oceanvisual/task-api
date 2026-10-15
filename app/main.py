import time
import uuid
from contextlib import asynccontextmanager

import gradio as gr
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from starlette.middleware.base import BaseHTTPMiddleware

from app.agent.graph import build_agent_graph
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.rag.chain import build_rag_pipeline
from app.schemas.agent import AgentRequest, AgentResponse, TraceStep
from app.schemas.agent import Source as AgentSource
from app.schemas.chat import ChatRequest, ChatResponse, Source

log = get_logger("task_api.main")

# answer-цепочка принимает {"question", "docs"}: retrieval делаем один раз
# и отдаём те же документы и в LLM, и в панель источников.
_answer_chain = None
_retriever = None

# Агент (LangGraph) с памятью в MemorySaver — живёт рядом с RAG-цепочкой.
_agent_graph = None
_agent_checkpointer = None

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
    global _answer_chain, _retriever, _agent_graph, _agent_checkpointer
    configure_logging()
    _answer_chain, _retriever = build_rag_pipeline()
    _agent_checkpointer = MemorySaver()
    _agent_graph = build_agent_graph(checkpointer=_agent_checkpointer)
    log.info("startup_complete", component="rag+agent", model=settings.llm_model)
    yield
    _answer_chain = None
    _retriever = None
    _agent_graph = None
    _agent_checkpointer = None


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


def _provider_http_error(exc: Exception, endpoint: str) -> HTTPException:
    """Залогировать ошибку LLM-провайдера и собрать 503 для клиента."""
    status, message = _provider_error(exc)
    log.error(
        "llm_error",
        endpoint=endpoint,
        exc_type=type(exc).__name__,
        provider_status=status,
        detail=message,
    )
    detail = (
        "LLM provider temporarily unavailable. Try again in 30-60 seconds. "
        f"Raw: {type(exc).__name__}"
    )
    if status:
        detail += f" ({status})"
    if message:
        detail += f": {message}"
    return HTTPException(status_code=503, detail=detail)


class DisableProxyBufferingMiddleware(BaseHTTPMiddleware):
    """Говорит nginx и прочим прокси НЕ буферизовать SSE-стримы Gradio
    (X-Accel-Buffering: no). Без этого live-обновления статуса/токенов
    копятся в буфере прокси и прилетают одним пакетом в самом конце.

    Gradio смонтирован на /ui, поэтому его очередь — /ui/queue/{join,data};
    матчим по вхождению, чтобы заголовок навешивался с учётом префикса.
    """

    SSE_MARKERS = ("/queue/data", "/queue/join")

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if any(m in request.url.path for m in self.SSE_MARKERS):
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="RAG service", lifespan=lifespan)
app.add_middleware(DisableProxyBufferingMiddleware)


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
        raise _provider_http_error(exc, "/chat") from exc
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
            full_context=doc.page_content,
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


def _current_turn(messages: list) -> list:
    """Сообщения текущего хода: от последнего HumanMessage до конца.

    С thread_id граф возвращает всю историю треда из checkpointer; без среза
    в trace/sources/tools_used попадают прошлые ходы.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _extract_trace(messages: list) -> tuple[list[TraceStep], list[str]]:
    """Собрать ReAct-трейс из истории сообщений графа.

    На каждый tool_call в AIMessage создаём шаг; его output заполняем из
    следующего ToolMessage (output инициализируется плейсхолдером, пока тул
    не отработал).
    """
    steps: list[TraceStep] = []
    tools_used: list[str] = []
    step_num = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                step_num += 1
                tools_used.append(tc["name"])
                steps.append(
                    TraceStep(
                        step=step_num,
                        node="agent",
                        tool=tc["name"],
                        input=tc.get("args", {}),
                        output="(tool requested)",
                        latency_ms=0,
                    )
                )
        elif (
            isinstance(msg, ToolMessage)
            and steps
            and steps[-1].output == "(tool requested)"
        ):
            steps[-1].output = str(msg.content)[:500]
    return steps, tools_used


@app.post("/agent", response_model=AgentResponse)
def agent_chat(payload: AgentRequest) -> AgentResponse:
    try:
        check_input(payload.question)
    except GuardrailError as e:
        log.warning("guardrail_rejected", stage="input", reason=str(e))
        raise HTTPException(status_code=422, detail=f"Input rejected: {e}") from e

    thread_id = payload.thread_id or str(uuid.uuid4())
    # contextvars: привязываем thread_id ко всем логам этого запроса (в т.ч. из графа/тулов)
    structlog.contextvars.bind_contextvars(thread_id=thread_id)
    try:
        log.info("agent_request", question_len=len(payload.question))
        t0 = time.perf_counter()
        try:
            result = _agent_graph.invoke(
                {
                    "messages": [HumanMessage(content=payload.question)],
                    "iteration_count": 0,
                },
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            raise _provider_http_error(exc, "/agent") from exc
        total_ms = int((time.perf_counter() - t0) * 1000)

        # checkpointer возвращает всю историю треда — разбираем только текущий ход
        turn_messages = _current_turn(result["messages"])
        raw_answer = turn_messages[-1].content
        trace_steps, tools_used = _extract_trace(turn_messages)
        if trace_steps:
            trace_steps[-1].latency_ms = total_ms

        safe_answer, guardrail_reason = check_output(raw_answer, tools_used)
        log.info(
            "agent_response",
            iterations=result.get("iteration_count", 0),
            tools_used=tools_used,
            latency_ms=total_ms,
            guardrail=guardrail_reason,
            answer_len=len(safe_answer),
        )
    finally:
        structlog.contextvars.clear_contextvars()

    sources: list[AgentSource] = []
    for msg in turn_messages:
        if isinstance(msg, ToolMessage) and "Sources:" in str(msg.content):
            body, _, sources_block = str(msg.content).partition("Sources:")
            tool_text = body.strip()  # текст, который агент реально получил от тула
            for line in sources_block.strip().splitlines():
                url = line.strip().lstrip("- ").strip()
                if url:
                    sources.append(
                        AgentSource(
                            url=url,
                            snippet=tool_text[:200],
                            full_context=tool_text,
                        )
                    )

    return AgentResponse(
        answer=safe_answer,
        trace=trace_steps,
        sources=sources,
        guardrail_triggered=guardrail_reason,
        iterations=result.get("iteration_count", 0),
    )


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
        log.error(
            "llm_error",
            handler="respond",
            exc_type=type(exc).__name__,
            provider_status=status,
            detail=message,
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


def respond_agent(message: str, history: list, thread_id_state: str):
    """Streaming-обработчик режима «Агент».

    Yields на каждом событии графа: смена статуса (какой тул вызывается),
    токены финального ответа, итоговый трейс. Gradio дорисовывает UI на каждый
    yield. Кортеж из 7 значений — под outputs ниже.
    """
    if not message or not message.strip():
        yield (
            history,
            "",
            "_Пустой запрос_",
            "_—_",
            "_—_",
            "💤 Готов к работе",
            thread_id_state,
        )
        return

    try:
        check_input(message)
    except GuardrailError as e:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ Запрос отклонён guardrails: {e}"},
        ]
        yield (
            history,
            "",
            "_Отклонён guardrails_",
            "_—_",
            "_—_",
            "🛡 Guardrail отклонил запрос",
            thread_id_state,
        )
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    thread_id = thread_id_state or str(uuid.uuid4())

    t0 = time.perf_counter()
    yield (
        history,
        "",
        "_Старт…_",
        "_—_",
        "_—_",
        "🤔 LLM решает, какой инструмент вызвать…",
        thread_id,
    )

    accumulated = ""
    final_messages: list = []
    tool_calls_count = 0
    iterations = 0

    try:
        for stream_mode, payload in _agent_graph.stream(
            {"messages": [HumanMessage(content=message)], "iteration_count": 0},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["updates", "messages"],
        ):
            if stream_mode == "updates":
                # payload = {"agent": {...}} или {"tool_executor": {...}}
                node, state_update = next(iter(payload.items()))
                node_msgs = state_update.get("messages", []) or []
                if node_msgs:
                    final_messages = final_messages + node_msgs

                if node == "agent":
                    iterations += 1
                    last = node_msgs[-1] if node_msgs else None
                    tool_calls = getattr(last, "tool_calls", None) or []
                    if tool_calls:
                        tc = tool_calls[0]
                        name = tc["name"]
                        tool_calls_count += 1
                        status = {
                            "documentation_search": "📚 Ищу в документации scikit-learn…",
                            "python_repl": "🧮 Считаю в Python REPL…",
                            "web_search": "🌐 Ищу в интернете…",
                        }.get(name, f"🛠 Вызываю {name}…")
                        yield (
                            history,
                            "",
                            "_В процессе…_",
                            "_—_",
                            "_—_",
                            status,
                            thread_id,
                        )
                elif node == "tool_executor":
                    tool_msg = node_msgs[-1] if node_msgs else None
                    tool_name = getattr(tool_msg, "name", "tool")
                    length = len(getattr(tool_msg, "content", "") or "")
                    status = f"🔍 Анализирую результат {tool_name} ({length} симв.), решаю что делать дальше…"
                    yield history, "", "_В процессе…_", "_—_", "_—_", status, thread_id

            elif stream_mode == "messages":
                # payload = (AIMessageChunk, metadata)
                msg_chunk, meta = payload
                chunk_text = getattr(msg_chunk, "content", "") or ""
                # только финальный ответ агента, без reasoning-промежутков tool-вызовов
                if (
                    chunk_text
                    and meta.get("langgraph_node") in ("agent", "finalize")
                    and not getattr(msg_chunk, "tool_call_chunks", None)
                ):
                    accumulated += chunk_text
                    history[-1]["content"] = accumulated
                    yield (
                        history,
                        "",
                        "_…_",
                        "_—_",
                        "_—_",
                        f"✍️ LLM пишет ответ потоком… {len(accumulated)} симв.",
                        thread_id,
                    )
    except Exception as exc:  # noqa: BLE001 — любую ошибку провайдера показываем в UI
        status_code, err = _provider_error(exc)
        log.error(
            "agent_error",
            handler="respond_agent",
            exc_type=type(exc).__name__,
            provider_status=status_code,
            detail=err,
        )
        history[-1]["content"] = (
            f"⚠️ Агент упал ({type(exc).__name__}). {err or ''}".strip()
        )
        yield history, "", "_Ошибка_", "_—_", "_—_", "❌ Ошибка агента", thread_id
        return

    total_ms = int((time.perf_counter() - t0) * 1000)

    trace_steps, tools_used = _extract_trace(final_messages)

    # fallback: если токены не стримились (LLM без streaming=True) — берём
    # финальный AIMessage из истории графа, иначе в чат уедет пустота.
    if not accumulated and final_messages:
        last_ai = next(
            (
                m
                for m in reversed(final_messages)
                if isinstance(m, AIMessage) and m.content
            ),
            None,
        )
        if last_ai:
            accumulated = str(last_ai.content)

    safe_answer, _ = check_output(accumulated, tools_used)
    history[-1]["content"] = safe_answer

    trace_md_lines = ["### Шаги агента", ""]
    for s in trace_steps:
        trace_md_lines.append(
            f"**{s.step}.** `{s.node}` → tool `{s.tool}` · "
            f"args `{s.input}` · output `{s.output[:120]}…`"
        )
    trace_md = "\n\n".join(trace_md_lines) if trace_steps else "_Без tool-вызовов_"

    timings = (
        f"### ⏱ Тайминги\n\n"
        f"- 🤖 **Total:** {total_ms} ms\n"
        f"- 🔄 **Итераций:** {iterations}\n"
        f"- 🛠 **Tool-вызовов:** {tool_calls_count}"
    )
    sources_panel = "### 📚 Источники\n\n_см. блок «Шаги агента»_"
    final_status = f"✅ Готово за {total_ms / 1000:.1f} сек · {tool_calls_count} tool-вызовов · {iterations} итераций"

    yield history, "", timings, sources_panel, trace_md, final_status, thread_id


def _route_respond(message: str, history: list, mode: str, thread_id_state: str):
    """Диспетчер между режимами. Оба пути отдают 7 значений под общий outputs."""
    if mode == "Агент (/agent)":
        yield from respond_agent(message, history, thread_id_state)
    else:
        # старый /chat handler отдаёт 4 значения — добиваем trace/status/thread
        for out in respond(message, history):
            yield (
                *out,
                "_(режим Быстрый — trace не используется)_",
                "✅ Готово",
                thread_id_state,
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
    status_line = gr.Markdown("💤 Готов к работе", elem_id="status-line")
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
            mode_radio = gr.Radio(
                choices=["Быстрый (/chat)", "Агент (/agent)"],
                value="Быстрый (/chat)",
                label="Режим",
            )
            timings_md = gr.Markdown(
                "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
            )
            sources_md = gr.Markdown("### 📚 Источники\n\n_—_")
            with gr.Accordion("Что сделал агент", open=False):
                trace_md = gr.Markdown("_Включите режим «Агент», чтобы увидеть шаги._")

    thread_id_state = gr.State("")

    inputs = [msg, chatbot, mode_radio, thread_id_state]
    outputs = [
        chatbot,
        msg,
        timings_md,
        sources_md,
        trace_md,
        status_line,
        thread_id_state,
    ]

    msg.submit(_route_respond, inputs, outputs)
    send.click(_route_respond, inputs, outputs)


demo.queue()
app = gr.mount_gradio_app(app, demo, path="/ui", css=CSS, theme=gr.themes.Soft())
