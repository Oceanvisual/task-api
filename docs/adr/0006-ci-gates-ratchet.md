# 0006. Гейты CI: ratchet для mypy, format только по изменённым файлам

- Дата: 2026-10-15
- Статус: Accepted
- Источник: PR #11

## Контекст
Деплой срабатывал на push в main без линтера и тайпчека; два workflow дублировали тесты.
В коде 21 ошибка mypy и неотформатированные файлы, затронутые незавершённой работой — массовый фикс дал бы конфликты.

## Решение
Один workflow: `lint` + `test` (mypy ratchet против `.mypy-baseline`, pytest) → `deploy`.
`ruff format --check` только по файлам, изменённым в PR. Локально то же самое — `scripts/check.sh`.

## Последствия
Число ошибок mypy только убывает; baseline обновляется `scripts/mypy-ratchet.sh --update` в PR, который их чинит.
Когда baseline = 0 — заменить ratchet на обычный `mypy app`; когда все файлы отформатированы — `ruff format --check .`.
