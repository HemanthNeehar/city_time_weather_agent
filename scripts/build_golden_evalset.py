"""
scripts/build_golden_evalset.py

Builds and validates a canonical Google ADK Golden Dataset (EvalSet)
from structured test definitions (mimicking SME tables or BigQuery trace exports).

Output: evalset1.evalset.json
"""
from pathlib import Path
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_case import EvalCase, Invocation, IntermediateData, SessionInput
from google.genai import types

def build_eval_case(
    eval_id: str,
    user_prompt: str,
    expected_tool_name: str,
    expected_tool_args: dict,
    reference_response: str,
    app_name: str = "city_time_weather_agent",
) -> EvalCase:
    """Constructs a strictly-typed ADK EvalCase for trajectory & response evaluation."""
    return EvalCase(
        eval_id=eval_id,
        conversation=[
            Invocation(
                invocation_id=f"inv-{eval_id}-1",
                user_content=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)],
                ),
                intermediate_data=IntermediateData(
                    tool_uses=[
                        types.FunctionCall(
                            name=expected_tool_name,
                            args=expected_tool_args,
                        )
                    ]
                ),
                final_response=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=reference_response)],
                ),
            )
        ],
        session_input=SessionInput(app_name=app_name, user_id="golden_eval_suite", state={}),
    )

def main():
    # Canonical cases representing SME rules or mined production traces
    raw_cases = [
        {
            "eval_id": "tokyo_weather",
            "prompt": "What is the weather in Tokyo?",
            "tool_name": "get_weather",
            "tool_args": {"city": "Tokyo"},
            "reference": "The weather in Tokyo is cloudy with a temperature of 18°C.",
        },
        {
            "eval_id": "london_time",
            "prompt": "What time is it in London right now?",
            "tool_name": "get_current_time",
            "tool_args": {"city": "London"},
            "reference": "The current time in London is 15:30:00 BST.",
        },
        {
            "eval_id": "paris_weather_and_time",
            "prompt": "Tell me the weather in Paris.",
            "tool_name": "get_weather",
            "tool_args": {"city": "Paris"},
            "reference": "The weather in Paris, France is clear sky with a temperature of 19.5°C.",
        },
    ]

    eval_cases = [
        build_eval_case(
            eval_id=c["eval_id"],
            user_prompt=c["prompt"],
            expected_tool_name=c["tool_name"],
            expected_tool_args=c["tool_args"],
            reference_response=c["reference"],
        )
        for c in raw_cases
    ]

    eval_set = EvalSet(
        eval_set_id="city_time_weather_golden_v1",
        eval_cases=eval_cases,
    )

    output_path = Path("evalset1.evalset.json")
    json_payload = eval_set.model_dump_json(indent=2)
    output_path.write_text(json_payload, encoding="utf-8")
    print(f"✅ Generated {len(eval_cases)} golden evaluation cases in: {output_path.resolve()}")

    # Defensive check: re-validate with Pydantic
    reloaded = EvalSet.model_validate_json(output_path.read_text(encoding="utf-8"))
    print(f"🔒 Schema validation passed! Loaded eval_set_id: '{reloaded.eval_set_id}'")

if __name__ == "__main__":
    main()

