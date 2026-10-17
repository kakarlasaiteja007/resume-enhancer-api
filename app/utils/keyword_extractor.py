"""Utilities for extracting resume-relevant keywords from job descriptions."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.services.llm_service import LLMService

KEYWORD_EXTRACTION_SYSTEM_PROMPT = """You extract hiring keywords from job descriptions.
Prioritize concrete skills, tools, certifications, role titles, domain knowledge, and
work methods that a candidate should mirror in a resume. Merge obvious duplicates,
preserve meaningful multi-word phrases, and exclude generic filler such as 'team player'
or 'excellent communication skills'.

For every keyword, assign an importance weight from 0.0 to 1.0. Base the score on how
explicitly the requirement is stated, whether it is required rather than preferred, and
how central it is to the role. Use higher scores for core required skills and lower scores
for optional or secondary qualifications. Return only a valid JSON object where each key
is a keyword and each value is its numeric weight. Do not include Markdown, explanations,
numbering, or an object wrapper beyond that JSON object."""


def extract_keywords(
    job_description: str,
    *,
    llm_service: LLMService | None = None,
) -> dict[str, float]:
    """Extract up to the configured number of weighted, resume-relevant keywords.

    Args:
        job_description: The full job description to analyze.
        llm_service: Optional service injection for callers that manage the LLM lifecycle.
    """
    if not job_description or not job_description.strip():
        return {}

    settings = get_settings()
    prompt = (
        f"Extract at most {settings.max_keywords} weighted keywords from this job description.\n\n"
        f"Job description:\n{job_description.strip()}"
    )
    service = llm_service or LLMService()
    json_chat_model = cast(
        ChatOpenAI,
        service.chat_model.bind(response_format={"type": "json_object"}),
    )
    response = service.invoke(
        prompt,
        system_prompt=KEYWORD_EXTRACTION_SYSTEM_PROMPT,
        chat_model=json_chat_model,
    )
    return parse_keyword_weights(response.content, settings.max_keywords)


def extract_job_keywords(
    job_description: str,
    *,
    llm_service: LLMService | None = None,
) -> dict[str, float]:
    """Extract weighted job keywords; kept for callers using the original function name."""
    return extract_keywords(job_description, llm_service=llm_service)


def parse_keyword_weights(content: Any, max_keywords: int) -> dict[str, float]:
    """Parse the model's weighted keyword JSON object and normalize its values."""
    raw_content = _extract_text_content(content)

    try:
        parsed = _load_json_object(raw_content)
    except json.JSONDecodeError as error:
        parsed = _parse_plaintext_keyword_weights(raw_content)
        if not parsed:
            raise ValueError(
                "The LLM response did not contain keyword weights in JSON or plain-text form."
            ) from error

    if not isinstance(parsed, dict):
        raise ValueError("The LLM keyword response must be a JSON object.")

    keywords: dict[str, float] = {}
    seen: set[str] = set()
    for item, weight in parsed.items():
        if not isinstance(item, str) or isinstance(weight, bool):
            continue
        keyword = item.strip()
        normalized = keyword.casefold()
        try:
            normalized_weight = round(min(max(float(weight), 0.0), 1.0), 2)
        except (TypeError, ValueError):
            continue
        if keyword and normalized not in seen:
            keywords[keyword] = normalized_weight
            seen.add(normalized)
        if len(keywords) == max_keywords:
            break
    return keywords


def _extract_text_content(content: Any) -> str:
    """Convert LangChain string or content-block responses into text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if text_parts:
            return "\n".join(text_parts).strip()
    return json.dumps(content).strip()


def _load_json_object(raw_content: str) -> Any:
    """Read a JSON object even when the model surrounds it with prose or code fences."""
    normalized_content = raw_content.strip()
    if normalized_content.startswith("```"):
        normalized_content = normalized_content.split("\n", maxsplit=1)[-1]
        normalized_content = normalized_content.rsplit("```", maxsplit=1)[0].strip()

    object_start = normalized_content.find("{")
    if object_start == -1:
        raise json.JSONDecodeError("No JSON object found", normalized_content, 0)

    return json.JSONDecoder().raw_decode(normalized_content[object_start:])[0]


def _parse_plaintext_keyword_weights(raw_content: str) -> dict[str, float]:
    """Parse common fallback formats such as ``Python: 0.95`` or ``Python (0.95)``."""
    pattern = re.compile(
        r"^\s*(?:[-*]|\d+[.)])?\s*['\"`]?(.+?)['\"`]?\s*"
        r"(?::|[-–—]|\()\s*(1(?:\.0+)?|0(?:\.\d+)?)\)?\s*$",
        re.MULTILINE,
    )
    return {keyword.strip(): float(weight) for keyword, weight in pattern.findall(raw_content)}
