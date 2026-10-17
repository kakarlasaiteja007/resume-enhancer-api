"""Settings loaded from an environment-specific YAML file and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class AppConfig(BaseModel):
    name: str = "Resume Enhancer API"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False


class Settings(BaseSettings):
    """Typed application settings.

    Values in a `.env` file or the process environment override YAML values.
    Nested YAML keys use double underscores when overridden, for example
    `APP_APP__DEBUG=true`.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="APP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: str = Field(default="development")
    app: AppConfig = Field(default_factory=AppConfig)
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "APP_OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-5-mini"
    openai_base_url: str | None = None
    openai_temperature: float = 0.2
    openai_max_tokens: int | None = 1_000
    openai_timeout: float = 60.0
    openai_max_retries: int = 2
    openai_reasoning_effort: str | None = "low"
    max_keywords: int = Field(default=10, ge=1)
    database_url: str = "sqlite:///./resume_enhancer.db"


def _load_yaml(environment: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / f"{environment}.yml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Set APP_ENV to development, test, or production."
        )
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


@lru_cache
def get_settings() -> Settings:
    environment = os.getenv("APP_ENV", "development")
    yaml_values = _load_yaml(environment)
    yaml_values["environment"] = environment
    return Settings(**yaml_values)
