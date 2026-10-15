# 0003. Весь langchain-стек на линии 0.3.x ради ragas 0.2.15

- Дата: 2026-10-06
- Статус: Accepted
- Источник: комментарии в requirements.txt; 25c2dc0

## Контекст
Снятие cap `<0.4` у langchain-community тянет langchain-core 1.x, на котором ragas 0.2.15 ломается в рантайме.
Широкие диапазоны давали ~30 минут backtracking в pip.

## Решение
langchain-core/openai/community/text-splitters, langgraph и langchain-experimental закреплены на ветках,
совместимых с core 0.3.x; `ragas==0.2.15`.

## Альтернативы
- Обновить ragas до версии под core 1.x: не проверено, затрагивает eval-ноутбук и метрики baseline.

## Последствия
Dependabot игнорирует minor/major для `langchain*`, `langgraph`, `ragas` (`.github/dependabot.yml`).
Пересмотреть: при переходе eval на ragas под core 1.x — один PR на весь стек + `/rag-eval`.
