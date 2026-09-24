# task-api: RAG-ассистент по документации scikit-learn

FastAPI + Gradio (`/ui`) + LangChain LCEL + Qdrant + `intfloat/multilingual-e5-small` + LLM через OpenRouter (OpenAI-совместимый API, `ChatOpenAI`). Прод: VPS за nginx, https://176-108-248-151.nip.io.

## Карта кода
- `app/main.py`: FastAPI (`/health`, `POST /chat`), Gradio UI со streaming (`respond`), lifespan собирает цепочку в глобальные `_chain`/`_retriever`.
- `app/rag/chain.py`: `SYSTEM_PROMPT`, `get_vectorstore()`, `build_rag_chain() -> (chain, retriever)`. Единственное место сборки RAG.
- `app/llm.py`: фабрика `get_llm()` (провайдер меняется только через settings).
- `app/config.py`: **канонический** `Settings` (pydantic-settings, `.env`). `app/core/config.py` только реэкспорт для совместимости, в нём ничего не добавлять.
- `app/scripts/load_corpus.py` → `data/corpus_chunks.jsonl`; `app/scripts/index_corpus.py` **пересоздаёт** коллекцию Qdrant.
- `notebooks/rag_eval.ipynb` → `notebooks/rag_metrics.json`: golden-датасет (10 вопросов), Recall@k + RAGAS 0.2.15.

## Жёсткие ограничения
- **Push/merge в `main` = деплой в прод** (`.github/workflows/deploy.yml`: build → GHCR → ssh на VPS). Только через PR из feature-ветки.
- **Прод и CI на Python 3.11** (Dockerfile, workflows). Локальный `.venv` на 3.13, поэтому не использовать синтаксис 3.12+ (`type X = ...`, PEP 695 generics `def f[T]()`, `@override` из typing и т.п.).
- **Зависимости: `requirements.txt`** (источник правды для Docker и CI; pyproject без deps). Torch ставится CPU-only через `--extra-index-url`, это не ломать. `langchain-community<0.4` и `ragas==0.2.15` зафиксированы намеренно.
- `Settings.llm_api_key` обязателен: без него падает импорт `app.config`. Тесты ставят dummy в `tests/conftest.py`.
- Новые настройки добавлять в `Settings` с дефолтом, а в `docker-compose.yml` пробрасывать через `${VAR:-default}`. Секреты только через env/GitHub secrets.

## Команды
Локально (пока нет миграции на uv-проект, глобальное правило «только uv» здесь адаптировано):
```bash
uv venv --python 3.11 .venv                                     # как в проде
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --only-binary sqlalchemy -r requirements.txt                   # см. примечание ниже
.venv/bin/python -m pytest -v                                   # тесты (без сети, всё замокано)
uvx ruff check app tests                                        # линт
docker network create mentoring-net 2>/dev/null; docker compose up -d qdrant
docker compose run --rm app sh -c "python -m app.scripts.load_corpus && python -m app.scripts.index_corpus"
docker compose up -d app          # http://127.0.0.1:8000/ui/
```
`--index-strategy unsafe-best-match` повторяет поведение pip с `--extra-index-url` (иначе uv берёт `requests` и др. только с индекса torch).
`--only-binary sqlalchemy`: sdist sqlalchemy 2.1.0 (транзитивно из langchain-community) не собирается.

## Тесты
- Все тесты патчат `app.main.build_rag_chain` (MagicMock chain/retriever). Реальные LLM/Qdrant/HF-модель в тестах запрещены.
- Для новой логики в `chain.py` тестировать чистые функции (`format_docs_with_sources` и т.п.) без поднятия цепочки.

## Изменения RAG (промпт, chunking, top_k, эмбеддер, модель)
После любого такого изменения прогнать `/rag-eval` и сравнить с `notebooks/rag_metrics.json`. Регрессия Recall@4 или Faithfulness означает, что изменение не мержим без обсуждения. Метрики в README обновлять из JSON (сейчас README и JSON расходятся).

## Известный техдолг (не чинить попутно без запроса, но учитывать)
- Retriever вызывается дважды на запрос: `_retriever.invoke()` для sources + внутри `_chain`. Двойной embed + Qdrant.
- `chat` и `respond` синхронные; логирование через `print`/stdlib вместо structlog; нет type hints у `build_rag_chain`, `lifespan`, `respond`.
- `gradio>=5.0.0` без верхней границы → ставится 6.x, где `theme`/`css` в `gr.Blocks(...)` игнорируются (UserWarning в тестах): кастомный CSS UI, вероятно, не применяется.
- `ci.yml` и `deploy.yml` оба гоняют тесты на PR и push в main (дублирование).
- `discount_calculator.py`, `app/index_corpus.py` (untracked, дубль `app/scripts/index_corpus.py`) и `wine-train/` к сервису не относятся.
