"""LangChain pipeline for matching candidate experience to a job description."""

from __future__ import annotations

from typing import Any

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from pydantic import BaseModel, Field

from app.services.llm_service import LLMService
from app.utils.keyword_extractor import extract_keywords


class CandidateKeywordScores(BaseModel):
    """Structured LLM response containing resume evidence scores by job keyword."""

    keyword_scores: dict[str, float] = Field(
        description="Candidate evidence scores from 0.0 to 1.0 for the requested keywords."
    )


CANDIDATE_KEYWORD_MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Assess how strongly a candidate resume demonstrates each requested job keyword.
Score only evidence explicitly stated in the resume. Use 1.0 for strong direct evidence,
0.7 for clear relevant experience, 0.4 for adjacent or partial experience, and 0.0 when
the resume provides no evidence. Use the exact requested keyword text as every key in
`keyword_scores`. {format_instructions}""",
        ),
        (
            "human",
            "Requested keywords: {keywords}\n\nCandidate resume:\n{candidate_resume_content}",
        ),
    ]
)


def calculate_candidate_experience_match(
    job_description: str,
    candidate_resume_content: str,
    *,
    llm_service: LLMService | None = None,
) -> dict[str, Any]:
    """Return a job-weighted candidate match percentage and per-keyword details.

    The pipeline calculates ``sum(job_weight * candidate_score) / sum(job_weight) * 100``.
    """
    service = llm_service or LLMService()
    match_chain = _build_match_chain(service)
    return match_chain.invoke(
        {
            "job_description": job_description,
            "candidate_resume_content": candidate_resume_content,
        }
    )


def _build_match_chain(service: LLMService) -> RunnableSequence:
    """Build the LCEL sequence used for extraction, scoring, and aggregation."""
    parser = PydanticOutputParser(pydantic_object=CandidateKeywordScores)
    score_chain = (
        RunnableLambda(_to_prompt_variables)
        | CANDIDATE_KEYWORD_MATCH_PROMPT.partial(
            format_instructions=parser.get_format_instructions()
        )
        | service.chat_model.bind(response_format={"type": "json_object"})
        | parser
    )
    score_and_calculate_chain = RunnableParallel(
        {
            "match_input": RunnablePassthrough(),
            "candidate_scores": score_chain,
        }
    ) | RunnableLambda(_calculate_weighted_match)

    return RunnableSequence(
        RunnableLambda(lambda values: _extract_job_keyword_weights(values, service)),
        RunnableBranch(
            (_has_no_job_keyword_weights, RunnableLambda(_empty_match_result)),
            (
                lambda values: not values["candidate_resume_content"].strip(),
                RunnableLambda(_empty_candidate_match_result),
            ),
            score_and_calculate_chain,
        ),
    )


def _extract_job_keyword_weights(
    values: Any,
    service: LLMService,
) -> dict[str, Any]:
    job_description = values.get("job_description") if isinstance(values, dict) else None
    if not isinstance(job_description, str):
        raise TypeError("The candidate match chain requires a string job_description.")

    job_keyword_weights = extract_keywords(job_description, llm_service=service)
    print(f"[Candidate Match] Extracted job keyword weights: {job_keyword_weights}")
    return {**values, "job_keyword_weights": job_keyword_weights}


def _has_no_job_keyword_weights(values: dict[str, Any]) -> bool:
    return not values["job_keyword_weights"]


def _to_prompt_variables(values: dict[str, Any]) -> dict[str, str]:
    return {
        "keywords": ", ".join(values["job_keyword_weights"]),
        "candidate_resume_content": values["candidate_resume_content"].strip(),
    }


def _calculate_weighted_match(values: dict[str, Any]) -> dict[str, Any]:
    match_input = values["match_input"]
    job_keyword_weights: dict[str, float] = match_input["job_keyword_weights"]
    structured_scores: CandidateKeywordScores = values["candidate_scores"]
    raw_scores = structured_scores.keyword_scores
    print(f"[Candidate Match] Raw LLM keyword scores: {raw_scores}")

    candidate_keyword_scores = {
        keyword: _normalize_score(raw_scores.get(keyword, 0.0)) for keyword in job_keyword_weights
    }
    print(f"[Candidate Match] Candidate keyword scores: {candidate_keyword_scores}")

    keyword_matches = [
        {
            "keyword": keyword,
            "job_weight": job_weight,
            "candidate_match_weight": candidate_keyword_scores[keyword],
        }
        for keyword, job_weight in job_keyword_weights.items()
    ]
    print(f"[Candidate Match] Per-keyword match details: {keyword_matches}")

    total_job_weight = sum(job_keyword_weights.values())
    weighted_score = sum(
        job_weight * candidate_keyword_scores[keyword]
        for keyword, job_weight in job_keyword_weights.items()
    )
    overall_match_percentage = round(weighted_score / total_job_weight * 100, 2)
    print(f"[Candidate Match] Total job keyword weight: {total_job_weight}")
    print(f"[Candidate Match] Weighted candidate score: {weighted_score}")
    print(f"[Candidate Match] Overall match percentage: {overall_match_percentage}%")

    return {
        "overall_match_percentage": overall_match_percentage,
        "job_keyword_weights": job_keyword_weights,
        "candidate_keyword_scores": candidate_keyword_scores,
        "keyword_matches": keyword_matches,
    }


def _empty_candidate_match_result(values: dict[str, Any]) -> dict[str, Any]:
    print("[Candidate Match] Resume content is empty; assigning 0.0 to every keyword.")
    keyword_weights = values["job_keyword_weights"]
    return _calculate_weighted_match(
        {
            "match_input": values,
            "candidate_scores": CandidateKeywordScores(
                keyword_scores={keyword: 0.0 for keyword in keyword_weights}
            ),
        }
    )


def _empty_match_result(_: dict[str, Any]) -> dict[str, Any]:
    print("[Candidate Match] No job keywords found; returning a 0.0% match.")
    return {
        "overall_match_percentage": 0.0,
        "job_keyword_weights": {},
        "candidate_keyword_scores": {},
        "keyword_matches": [],
    }


def _normalize_score(value: float) -> float:
    return round(min(max(float(value), 0.0), 1.0), 2)
