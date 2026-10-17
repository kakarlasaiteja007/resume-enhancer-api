"""Reusable LangChain wrapper for OpenAI chat models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import PROJECT_ROOT, get_settings

# Load local development secrets before creating settings or an LLM client.
# In deployed environments, the platform-provided environment variables are used instead.
load_dotenv(PROJECT_ROOT / ".env")


class LLMService:
    """Centralizes creation and invocation of the configured ChatOpenAI model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._chat_model: ChatOpenAI | None = None

    @property
    def chat_model(self) -> ChatOpenAI:
        """Return a lazily-created ChatOpenAI client shared by this service instance."""
        if self._chat_model is None:
            if not self.settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required to invoke an OpenAI chat model. "
                    "Add it to .env or your deployment environment."
                )

            self._chat_model = ChatOpenAI(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                base_url=self.settings.openai_base_url,
                temperature=self.settings.openai_temperature,
                max_tokens=self.settings.openai_max_tokens,
                timeout=self.settings.openai_timeout,
                max_retries=self.settings.openai_max_retries,
                reasoning_effort=self.settings.openai_reasoning_effort,
            )
        return self._chat_model

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        chat_model: ChatOpenAI | None = None,
    ) -> AIMessage:
        """Synchronously send a prompt and return the model response."""
        return self.invoke_messages(
            self._build_messages(prompt, system_prompt),
            chat_model=chat_model,
        )

    async def ainvoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        chat_model: ChatOpenAI | None = None,
    ) -> AIMessage:
        """Asynchronously send a prompt and return the model response."""
        return await self.ainvoke_messages(
            self._build_messages(prompt, system_prompt),
            chat_model=chat_model,
        )

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        chat_model: ChatOpenAI | None = None,
    ) -> AIMessage:
        """Synchronously invoke the model with pre-built LangChain messages."""
        # ChatOpenAI produces an AIMessage, while LangChain's base type is BaseMessage.
        model = chat_model or self.chat_model
        return cast(AIMessage, model.invoke(list(messages)))

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        chat_model: ChatOpenAI | None = None,
    ) -> AIMessage:
        """Asynchronously invoke the model with pre-built LangChain messages."""
        # ChatOpenAI produces an AIMessage, while LangChain's base type is BaseMessage.
        model = chat_model or self.chat_model
        return cast(AIMessage, await model.ainvoke(list(messages)))

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str | None) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages
