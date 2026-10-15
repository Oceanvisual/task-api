from langchain_core.tools import tool

from app.logging_config import get_logger
from app.rag.chain import build_rag_pipeline

log = get_logger("task_api.agent.tools")

_answer_chain = None
_retriever = None


def _ensure_rag_ready() -> None:
    global _answer_chain, _retriever
    if _answer_chain is None:
        # build_rag_pipeline -> (answer_chain, retriever): retrieval делаем сами,
        # один раз, и те же docs идут и в ответ, и в источники (как в app/main.py).
        _answer_chain, _retriever = build_rag_pipeline()


@tool
def documentation_search(query: str) -> str:
    """Search the scikit-learn documentation corpus for a given query.

    Use this tool when the user asks about scikit-learn classes, methods,
    parameters, or general ML concepts (Ridge, Lasso, decision trees, metrics).

    Args:
        query: natural-language question or keyword search.

    Returns:
        Answer text followed by a list of source URLs.
    """
    _ensure_rag_ready()
    docs = _retriever.invoke(query)
    answer = _answer_chain.invoke({"question": query, "docs": docs})
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    sources_block = "\n".join(f"- {url}" for url in sources)
    log.info("tool_call", tool="documentation_search", query=query, n_docs=len(docs))
    return f"{answer}\n\nSources:\n{sources_block}"


from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool

from app.config import settings

_python_repl = PythonREPLTool()
_ddg_search = DuckDuckGoSearchRun()

# SECURITY NOTE: PythonREPLTool executes arbitrary code in the same process
# as our app, and the code is written by the LLM on the user's request —
# i.e. prompt -> RCE (env secrets, os.system). Off by default
# (settings.enable_python_repl); for untrusted users use a sandbox
# (gvisor, firecracker, e2b.dev).


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the printed output.

    Use this tool for arithmetic, computing formulas, or transforming data.
    The code runs in a sandboxed REPL; use `print()` to surface results.

    Args:
        code: Python source code to execute.

    Returns:
        Stdout of the executed code, or error message.
    """
    if not settings.enable_python_repl:
        log.info("tool_call", tool="python_repl", code_len=len(code), disabled=True)
        return (
            "Python REPL is disabled in this environment. "
            "Compute the answer yourself and show the calculation steps."
        )
    log.info("tool_call", tool="python_repl", code_len=len(code))
    return _python_repl.run(code)


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo for recent or general-knowledge info.

    Use this tool when the question requires fresh data (release notes,
    latest versions, news) that is NOT in the documentation corpus.

    Args:
        query: free-text search query.

    Returns:
        Top-3 search results as title + snippet text.
    """
    if not settings.enable_web_search:
        log.info("tool_call", tool="web_search", query=query, disabled=True)
        return "Web search is disabled in this environment."
    log.info("tool_call", tool="web_search", query=query)
    return _ddg_search.run(query)


TOOLS = [documentation_search, python_repl, web_search]
