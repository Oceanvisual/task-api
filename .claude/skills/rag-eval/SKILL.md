---
name: rag-eval
description: Перезамерить качество RAG task-api (Recall@4 + RAGAS Faithfulness/Relevancy) через notebooks/rag_eval.ipynb и сравнить с baseline в rag_metrics.json. Использовать после изменений промпта, chunking, top_k, эмбеддера, LLM или корпуса.
argument-hint: "[--reindex — пересобрать корпус и индекс перед замером]"
---

# RAG eval (task-api)

Baseline (текущий в git): !`git show HEAD:notebooks/rag_metrics.json 2>/dev/null || echo "нет baseline в HEAD"`

## Предусловия
1. Qdrant поднят: `curl -s http://127.0.0.1:6333/collections`. Если нет, то `docker network create mentoring-net 2>/dev/null; docker compose up -d qdrant`. Docker daemon не запущен: попроси пользователя открыть Docker Desktop.
2. В `.env` есть `LLM_API_KEY` (не читать файл, просто проверить запуском: ноутбук упадёт с понятной ошибкой).
3. Jupyter в venv (однократно): `uv pip install --python .venv/bin/python nbconvert ipykernel`.
4. Локально `QDRANT_URL` должен указывать на `http://localhost:6333` (дефолт в Settings `http://qdrant:6333` работает только внутри docker-сети). Если в `.env` иначе, передай env при запуске: `QDRANT_URL=http://localhost:6333 ...`.

## Прогон
- Если в аргументах `--reindex` или менялся chunking/эмбеддер/корпус:
  `QDRANT_URL=http://localhost:6333 .venv/bin/python -m app.scripts.load_corpus && QDRANT_URL=http://localhost:6333 .venv/bin/python -m app.scripts.index_corpus`
- Eval (исполняет ноутбук на месте, перезаписывает `notebooks/rag_metrics.json`):
  ```bash
  QDRANT_URL=http://localhost:6333 .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 notebooks/rag_eval.ipynb
  ```
  RAGAS-судья ходит в LLM: это ~30–60 платных вызовов, прогон занимает несколько минут.

## Отчёт
Таблица baseline → new → Δ по `recall_at_4`, `faithfulness`, `answer_relevancy`. Пометь регрессии (Recall упал хотя бы на 1 hit, или Faithfulness/Relevancy упали больше чем на 0.03; при n=10 шум LLM-судьи ±0.03–0.05, так что пограничные изменения перепроверь повторным прогоном).
Если метрики стали каноническими (изменение принято), предложи обновить таблицу в README из JSON.
Не коммить изменённый ноутбук с output без запроса: diff на сотни строк.
