"""
custom_metrics.py

Day 8: Domain-Specific Custom Code Metrics (EvalConfig.custom_metrics).
Deterministic Python guardrails that enforce binary business invariants
alongside ADK's built-in LLM-as-a-Judge rubrics.
"""

import re
from typing import Optional

from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import Invocation, get_all_tool_responses
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult


_TEMP_PATTERN = re.compile(r"\b-?\d+(?:\.\d+)?\s*(?:°\s*[CF]|degrees?\s+(?:celsius|fahrenheit))\b", re.IGNORECASE)
_REFUSAL_PHRASES = (
    "no weather",
    "cannot find",
    "could not find",
    "unable to find",
    "not available",
    "unsupported",
    "don't have weather",
    "do not have weather",
)


def _extract_final_text(inv: Invocation) -> str:
    """Extracts plain text from an Invocation's final_response."""
    if not inv.final_response or not inv.final_response.parts:
        return ""
    return " ".join(part.text for part in inv.final_response.parts if part.text).strip()


def _weather_tool_outcomes(inv: Invocation) -> tuple[bool, bool]:
    """Returns (any_error, any_success) across get_weather responses in this turn."""
    if not inv.intermediate_data:
        return False, False
    any_error = False
    any_success = False
    for tool_resp in get_all_tool_responses(inv.intermediate_data):
        if tool_resp.name == "get_weather" and isinstance(tool_resp.response, dict):
            status = tool_resp.response.get("status")
            if status == "error":
                any_error = True
            elif status == "success":
                any_success = True
    return any_error, any_success


def unsupported_city_guardrail(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]] = None,
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
    """Fail-closed guardrail: never output a temperature when weather lookup failed.

    Args:
        eval_metric: Metric metadata and threshold configuration from eval_config.json.
        actual_invocations: List of Invocation turns executed by the agent.
        expected_invocations: Optional golden invocations (unused by this reference-free check).
        conversation_scenario: Optional multi-turn scenario context.

    Returns:
        EvaluationResult with per-invocation scores and overall pass/fail status.
    """
    del expected_invocations, conversation_scenario
    threshold = eval_metric.threshold if eval_metric.threshold is not None else 1.0

    if not actual_invocations:
        return EvaluationResult(
            overall_score=1.0,
            overall_eval_status=EvalStatus.PASSED,
            per_invocation_results=[],
        )

    per_invocation_results: list[PerInvocationResult] = []
    for inv in actual_invocations:
        text = _extract_final_text(inv)
        text_lower = text.lower()
        has_temp = bool(_TEMP_PATTERN.search(text))
        tool_errored, tool_succeeded = _weather_tool_outcomes(inv)
        admitted_missing = any(phrase in text_lower for phrase in _REFUSAL_PHRASES)

        # Violation: weather lookup failed (with no valid weather payload returned) yet response cites a temperature
        violated = (tool_errored or admitted_missing) and not tool_succeeded and has_temp
        score = 0.0 if violated else 1.0
        status = EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED

        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=inv,
                score=score,
                eval_status=status,
            )
        )

    overall_score = sum(r.score for r in per_invocation_results if r.score is not None) / len(per_invocation_results)
    overall_status = EvalStatus.PASSED if overall_score >= threshold else EvalStatus.FAILED

    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=overall_status,
        per_invocation_results=per_invocation_results,
    )
