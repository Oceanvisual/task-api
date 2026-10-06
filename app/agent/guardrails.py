"""Лёгкие guardrails для агента: проверка входа и санитизация выхода.

Вход (`check_input`) — дешёвый детерминированный барьер против prompt-injection,
подмены роли, chat-template токенов и мусорных символов (emoji) до траты токенов.
Выход (`check_output`) — анти-галлюцинация: если ответ делает documentation-grounded
утверждения ("according to the docs"), но агент не вызывал documentation_search,
помечаем это. Возвращает причину срабатывания для телеметрии/ответа.

Это НЕ полноценная защита от jailbreak — только первый барьер.
"""

import re

from app.config import settings

_MAX_INPUT_CHARS = 500

# Попытки перезаписать системный промпт / сменить роль.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(the\s+)?(system|previous)\b",
    r"forget\s+(everything|all)\b",
    r"you\s+are\s+now\b",
    # «act as a» само по себе — обычная ML-лексика («L1 acts as a feature
    # selector»), поэтому ловим только смену роли/снятие ограничений.
    r"act\s+as\s+if\s+you\s+(are|were)\b",
    r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken|uncensored|evil|dan)\b",
    r"reveal\s+(your\s+)?(system|prompt|instructions|secrets?)",
    r"system\s*prompt",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Chat-template спец-токены (<|im_start|>, <|system|> и т.п.) — попытка
# инъекции на уровне токенизатора.
_CHAT_TEMPLATE_RE = re.compile(r"<\|.*?\|>")

# Emoji / пиктограммы: в техническом вопросе про sklearn им не место, а в
# payload-инъекциях они частый шум. Латиницу и кириллицу (мультиязычность) НЕ трогаем.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs (supplemental), вкл. 🚀
    "\U0001f000-\U0001f0ff"  # mahjong/dominoes/cards
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f1e6-\U0001f1ff"  # regional indicators (флаги)
    "]"
)

# Ответ ссылается на документацию ("according to the docs/documentation").
_DECLARATIVE_SOURCE_RE = re.compile(
    r"according to (the )?(documentation|docs)\b", re.IGNORECASE
)


class GuardrailError(Exception):
    """Вход отклонён guardrails (невалидный или подозрительный запрос)."""


def check_input(question: str) -> None:
    """Проверить вопрос пользователя. Бросает GuardrailError при проблеме."""
    text = (question or "").strip()
    if not text:
        raise GuardrailError("empty question")
    if len(text) > _MAX_INPUT_CHARS:
        raise GuardrailError(f"question too long (>{_MAX_INPUT_CHARS} chars)")
    if _CHAT_TEMPLATE_RE.search(text):
        raise GuardrailError("chat-template tokens are not allowed")
    if _EMOJI_RE.search(text):
        raise GuardrailError("emoji / pictographs are not allowed")
    if _INJECTION_RE.search(text):
        raise GuardrailError("possible prompt injection detected")


def check_output(answer: str, trace_tools: list[str]) -> tuple[str, str | None]:
    """Санитизировать ответ агента.

    Возвращает (safe_answer, reason): reason != None, если guardrail сработал.
    - documentation-grounded утверждение без вызова documentation_search
      -> "declarative_without_source" (возможная галлюцинация);
    - пустой ответ -> "empty_output";
    - превышение лимита длины -> "output_truncated_to_N".
    В норме reason=None и answer возвращается без изменений.
    """
    text = (answer or "").strip()
    if not text:
        return (
            "Не удалось сформировать ответ. Переформулируй вопрос.",
            "empty_output",
        )

    if _DECLARATIVE_SOURCE_RE.search(text) and "documentation_search" not in trace_tools:
        warning = (
            "⚠️ Ответ содержит documentation-grounded утверждения, но инструмент "
            "documentation_search не вызывался — возможна галлюцинация. "
            "Проверь источники.\n\n"
        )
        return warning + text, "declarative_without_source"

    limit = settings.agent_max_output_chars
    if len(text) > limit:
        return text[:limit].rstrip() + "…", f"output_truncated_to_{limit}"

    return text, None
