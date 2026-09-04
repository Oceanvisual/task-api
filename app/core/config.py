"""Обратная совместимость: канонический конфиг — app.config."""

from app.config import Settings, settings

__all__ = ["Settings", "settings"]
