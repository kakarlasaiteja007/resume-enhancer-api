"""Helpers for retrieving values from the active application configuration."""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings


def get_config_value(key: str, default: Any = None) -> Any:
    """Return a configuration value by key, or ``default`` when it is unavailable.

    Nested settings can be addressed with dots, for example ``app.name`` or
    ``app.api_prefix``.
    """
    if not key or not key.strip():
        return default

    value: Any = get_settings().model_dump()
    for key_part in key.strip().split("."):
        if not isinstance(value, dict) or key_part not in value:
            return default
        value = value[key_part]
    return value
