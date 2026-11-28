"""
tests/unit/test_custom_metrics.py

Gate 0 ($0 deterministic) unit tests for Day 8 `custom_metrics.unsupported_city_guardrail`.
Verifies fail-closed temperature fabrication rules without making live LLM calls.
"""

import pytest
from google.adk.evaluation.eval_case import Invocation, InvocationEvent, InvocationEvents
from google.adk.evaluation.eval_config import get_eval_metrics_from_config, get_evaluation_criteria_or_default
from google.adk.evaluation.eval_metrics import EvalStatus
from google.genai import types

from city_time_weather_agent.custom_metrics import unsupported_city_guardrail


def _guardrail_metric():
    cfg = get_evaluation_criteria_or_default("eval_config.json")
    metrics = {m.metric_name: m for m in get_eval_metrics_from_config(cfg)}
    return metrics["unsupported_city_guardrail"]


def _make_invocation(prompt: str, final_text: str, weather_responses: list[dict]) -> Invocation:
    events = [
        InvocationEvent(
            author="city_time_weather_agent",
            content=types.Content(
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="get_weather",
                            response=resp,
                        )
                    )
                ]
            ),
        )
        for resp in weather_responses
    ]
    return Invocation(
        user_content=types.Content(parts=[types.Part(text=prompt)]),
        final_response=types.Content(parts=[types.Part(text=final_text)]),
        intermediate_data=InvocationEvents(invocation_events=events),
    )


@pytest.mark.unit
class TestCustomMetricsGuardrail:
    def test_clean_refusal_passes(self):
        inv = _make_invocation(
            "What is the weather in Gloomsberg?",
            "Sorry, I could not find weather data for Gloomsberg.",
            [{"status": "error", "error_message": "No weather data for 'Gloomsberg'."}],
        )
        res = unsupported_city_guardrail(_guardrail_metric(), [inv])
        assert res.overall_score == 1.0
        assert res.overall_eval_status == EvalStatus.PASSED

    def test_fabricated_temperature_fails_closed(self):
        inv = _make_invocation(
            "What is the weather in Gloomsberg?",
            "I could not find Gloomsberg, but it is probably around 24°C today.",
            [{"status": "error", "error_message": "No weather data for 'Gloomsberg'."}],
        )
        res = unsupported_city_guardrail(_guardrail_metric(), [inv])
        assert res.overall_score == 0.0
        assert res.overall_eval_status == EvalStatus.FAILED

    def test_mixed_multi_tool_turn_passes_when_valid_city_succeeded(self):
        """Regression test: Xanadu succeeded (4.6°C) + NotACity123 errored in the same turn."""
        inv = _make_invocation(
            "Check Xanadu and NotACity123.",
            "Xanadu is 4.6°C. There is no weather data for NotACity123.",
            [
                {"status": "success", "report": "The weather in Xanadu is clear sky at 4.6°C."},
                {"status": "error", "error_message": "No weather data for 'NotACity123'."},
            ],
        )
        res = unsupported_city_guardrail(_guardrail_metric(), [inv])
        assert res.overall_score == 1.0
        assert res.overall_eval_status == EvalStatus.PASSED

    def test_normal_weather_report_passes(self):
        inv = _make_invocation(
            "What is the weather in Tokyo?",
            "The weather in Tokyo is cloudy with a temperature of 18°C.",
            [{"status": "success", "report": "cloudy, 18°C"}],
        )
        res = unsupported_city_guardrail(_guardrail_metric(), [inv])
        assert res.overall_score == 1.0
        assert res.overall_eval_status == EvalStatus.PASSED
