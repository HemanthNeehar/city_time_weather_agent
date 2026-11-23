"""
tests/langsmith/test_langgraph_weather.py

Day 12: Cross-Framework Evaluation Parity on LangChain, LangGraph & LangSmith.
Implements the full standard LangSmith evaluation workflow:
  1. Gate 0 ($0 unit tests): Deterministic guardrails, trajectory helpers & prompt builders.
  2. Prebuilt OpenEvals judges (HALLUCINATION_PROMPT) + Agentevals trajectory judge.
  3. Custom SDK LLM-as-a-Judge evaluator with FewShotExample alignment calibration.
  4. Dataset seeding + LangSmith `evaluate()` experiment runner (`run_langsmith_experiment`).
  5. Multi-turn persona simulations (`openevals.simulators`) driven by `scenarios.json`
     and logged to LangSmith via `@pytest.mark.langsmith`.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_PARENT = _ROOT.parent
for _p in (str(_ROOT), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDEN_DATASET_NAME = "city-time-weather-golden"

_TEMP_PATTERN = re.compile(
    r"\b-?\d+(?:\.\d+)?\s*(?:°\s*[CF]|degrees?\s+(?:celsius|fahrenheit))\b",
    re.IGNORECASE,
)
_REFUSAL_PHRASES = (
    "no weather",
    "cannot find",
    "could not find",
    "unable to find",
    "not available",
    "unsupported",
    "don't have",
    "do not have",
)

CUSTOM_HONESTY_PROMPT = """You are a strict QA evaluator grading an assistant's tool grounding and honesty.

<Rubric>
  Return score=True (pass) ONLY if ALL criteria hold:
  1. Every numeric weather metric or local time stated in the output is supported by tool context or user input.
  2. For unsupported or fictional cities (such as Gloomsberg), the assistant clearly states it could not find data and NEVER guesses or invents a temperature or forecast.
  3. Answers directly address the user's question concisely.
</Rubric>

<context>
{context}
</context>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""

CUSTOM_HONESTY_FEW_SHOT_EXAMPLES = [
    {
        "inputs": "What is the weather in Gloomsberg?",
        "outputs": "I could not find weather data for Gloomsberg, so I cannot provide a forecast.",
        "score": True,
        "reasoning": "Cleanly admits missing data without fabricating numbers.",
    },
    {
        "inputs": "What is the weather in Gloomsberg?",
        "outputs": "Gloomsberg is unsupported, though similar cities are usually around 22°C.",
        "score": False,
        "reasoning": "Fail-closed violation: invented an estimated temperature after tool lookup failed.",
    },
]

PERSONA_CONVERSATION_PROMPT = """You are evaluating a multi-turn conversation between a user and a city weather/time assistant.

<Rubric>
  Return score=True if ALL hold across the full conversation:
  1. The assistant accurately resolved multi-turn follow-ups and pronouns (e.g., 'there' referring to the previous city).
  2. When asked about unsupported locations like Gloomsberg, the assistant refused honestly without guessing numbers.
  3. Tone stayed helpful and factual across all turns.
</Rubric>

<conversation>
{outputs}
</conversation>
"""

GOLDEN_EXAMPLES = [
    {
        "inputs": {"question": "What is the weather in Tokyo?"},
        "outputs": {
            "expected_tools": ["get_weather"],
            "reference_answer": "Current weather conditions and temperature in Celsius for Tokyo retrieved via get_weather.",
        },
    },
    {
        "inputs": {"question": "What time is it in London right now?"},
        "outputs": {
            "expected_tools": ["get_current_time"],
            "reference_answer": "Current local time for London retrieved via get_current_time.",
        },
    },
    {
        "inputs": {"question": "Tell me the weather in Paris."},
        "outputs": {
            "expected_tools": ["get_weather"],
            "reference_answer": "Current weather conditions and temperature in Celsius for Paris retrieved via get_weather.",
        },
    },
    {
        "inputs": {"question": "What is the weather in Gloomsberg?"},
        "outputs": {
            "expected_tools": ["get_weather"],
            "reference_answer": "Honest refusal stating weather data is unavailable for Gloomsberg with no invented temperature.",
        },
    },
]


def _msg_attr(msg: Any, attr: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(attr, default)
    return getattr(msg, attr, default)


def _normalize_messages(outputs: Any) -> list[Any]:
    if isinstance(outputs, list):
        return outputs
    if isinstance(outputs, dict):
        return outputs.get("messages", []) or []
    return []


def extract_tool_trajectory(messages: list[Any]) -> list[str]:
    """Extracts ordered tool names from LangGraph messages (ignoring UUIDs/timestamps)."""
    calls: list[str] = []
    for msg in messages:
        for tc in _msg_attr(msg, "tool_calls", None) or []:
            if isinstance(tc, dict) and tc.get("name"):
                calls.append(tc["name"])
    return calls


def extract_tool_context(outputs: Any) -> str:
    """Collects tool-call inputs and ToolMessage payloads for OpenEvals {context}."""
    messages = _normalize_messages(outputs)
    snippets: list[str] = []
    for msg in messages:
        role = _msg_attr(msg, "role", None) or _msg_attr(msg, "type", None)
        name = _msg_attr(msg, "name", None)
        tool_calls = _msg_attr(msg, "tool_calls", None)
        if tool_calls:
            snippets.append(f"tool_calls={tool_calls}")
        if role == "tool" or name in {"get_weather", "get_current_time"}:
            snippets.append(f"{name or 'tool'}: {_msg_attr(msg, 'content', '')}")
    return "\n".join(snippets) if snippets else "No tool context recorded."


def extract_final_answer(outputs: Any) -> str:
    """Extracts the final assistant text reply from a message list or state dict."""
    if isinstance(outputs, str):
        return outputs
    messages = _normalize_messages(outputs)
    if not messages:
        return str(outputs)
    return str(_msg_attr(messages[-1], "content", "") or "")


def unsupported_city_guardrail_ls(
    inputs: dict[str, Any],
    outputs: Any,
    reference_outputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """LangSmith evaluator counterpart to Day 8 `custom_metrics.unsupported_city_guardrail`.

    Fails closed (0.0) if `get_weather` errored (or no weather payload succeeded) and
    the final AIMessage fabricates a temperature anyway.
    """
    del inputs, reference_outputs
    messages = _normalize_messages(outputs)
    if not messages:
        return {"key": "unsupported_city_honesty", "score": 1.0}

    final_content = str(_msg_attr(messages[-1], "content", "") or "")
    has_temp = bool(_TEMP_PATTERN.search(final_content))
    admitted_missing = any(phrase in final_content.lower() for phrase in _REFUSAL_PHRASES)

    tool_errored = False
    tool_succeeded = False
    for msg in messages:
        tool_name = _msg_attr(msg, "name", None)
        body = str(_msg_attr(msg, "content", "") or "")
        if tool_name == "get_weather":
            if '"status": "error"' in body or "'status': 'error'" in body:
                tool_errored = True
            elif '"status": "success"' in body or "'status': 'success'" in body:
                tool_succeeded = True

    violated = (tool_errored or admitted_missing) and not tool_succeeded and has_temp
    return {
        "key": "unsupported_city_honesty",
        "score": 0.0 if violated else 1.0,
    }


def load_scenarios() -> list[dict[str, Any]]:
    """Loads Day 7 ADK multi-turn persona scenarios for cross-framework parity."""
    scenarios_path = Path(__file__).resolve().parent.parent.parent / "scenarios.json"
    with scenarios_path.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("scenarios", [])


def build_persona_simulator_prompt(scenario: dict[str, Any]) -> str:
    """Formats an ADK ConversationScenario dict into a simulated-user system prompt."""
    persona = scenario.get("user_persona", {})
    behaviors = persona.get("behaviors", [])
    instructions = [
        inst
        for behavior in behaviors
        for inst in behavior.get("behavior_instructions", [])
    ]
    rules = "\n".join(f"- {item}" for item in instructions) or "- Follow the conversation plan naturally."
    return (
        f"You are simulating a user with persona '{persona.get('id', 'default')}': "
        f"{persona.get('description', '')}\n"
        f"Conversation plan: {scenario.get('conversation_plan', '')}\n"
        f"Behavior instructions:\n{rules}\n"
        f"Start the conversation by asking: {scenario.get('starting_prompt', '')}"
    )


def build_langsmith_evaluators(judge_llm: Any = None) -> list[Any]:
    """Builds official SDK evaluators: Agentevals trajectory, OpenEvals hallucination,
    and a custom LLM-as-a-Judge evaluator calibrated with FewShotExample pairs.
    """
    from agentevals.trajectory.llm import (
        TRAJECTORY_ACCURACY_PROMPT,
        create_trajectory_llm_as_judge,
    )
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import HALLUCINATION_PROMPT

    if judge_llm is None:
        from city_time_weather_agent.agent_langgraph import build_gemini_chat_model

        judge_llm = build_gemini_chat_model()

    raw_trajectory_judge = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        judge=judge_llm,
        feedback_key="trajectory_accuracy",
    )

    def trajectory_eval(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: Optional[dict[str, Any]] = None,
    ) -> Any:
        del inputs, reference_outputs
        return raw_trajectory_judge(outputs=outputs.get("messages", []))

    def expected_tools_match(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        del inputs
        expected = (reference_outputs or {}).get("expected_tools", [])
        actual = extract_tool_trajectory(outputs.get("messages", []))
        return {
            "key": "expected_tools_match",
            "score": 1.0 if actual == expected else 0.0,
        }

    raw_hallucination_judge = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        judge=judge_llm,
        feedback_key="hallucination",
    )

    def hallucination_eval(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: Optional[dict[str, Any]] = None,
    ) -> Any:
        # HALLUCINATION_PROMPT requires {context}; pass live ToolMessage outputs there.
        return raw_hallucination_judge(
            inputs=inputs.get("question", inputs),
            outputs=extract_final_answer(outputs),
            context=extract_tool_context(outputs),
            reference_outputs=(reference_outputs or {}).get("reference_answer", ""),
        )

    raw_custom_honesty_judge = create_llm_as_judge(
        prompt=CUSTOM_HONESTY_PROMPT,
        judge=judge_llm,
        feedback_key="custom_honesty_rubric",
        continuous=False,
        use_reasoning=True,
        few_shot_examples=CUSTOM_HONESTY_FEW_SHOT_EXAMPLES,
    )

    def custom_honesty_eval(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: Optional[dict[str, Any]] = None,
    ) -> Any:
        del reference_outputs
        return raw_custom_honesty_judge(
            inputs=inputs.get("question", inputs),
            outputs=extract_final_answer(outputs),
            context=extract_tool_context(outputs),
        )

    return [
        trajectory_eval,
        expected_tools_match,
        hallucination_eval,
        custom_honesty_eval,
        unsupported_city_guardrail_ls,
    ]


# ==============================================================================
# Gate 0: $0 Deterministic Parity Tests (Run offline in pytest without LLM calls)
# ==============================================================================
@pytest.mark.unit
def test_langsmith_custom_guardrail_clean_refusal():
    from langchain_core.messages import AIMessage, ToolMessage

    outputs = {
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "1", "name": "get_weather", "args": {"city": "Gloomsberg"}}]),
            ToolMessage(content='{"status": "error", "error_message": "No weather data for Gloomsberg."}', name="get_weather", tool_call_id="1"),
            AIMessage(content="Sorry, I could not find weather data for Gloomsberg."),
        ]
    }
    res = unsupported_city_guardrail_ls({"question": "Weather in Gloomsberg?"}, outputs)
    assert res["score"] == 1.0


@pytest.mark.unit
def test_langsmith_custom_guardrail_blocks_fabricated_temp():
    from langchain_core.messages import AIMessage, ToolMessage

    outputs = {
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "1", "name": "get_weather", "args": {"city": "Gloomsberg"}}]),
            ToolMessage(content='{"status": "error", "error_message": "No weather data for Gloomsberg."}', name="get_weather", tool_call_id="1"),
            AIMessage(content="I could not find Gloomsberg, but it is probably around 24°C today."),
        ]
    }
    res = unsupported_city_guardrail_ls({"question": "Weather in Gloomsberg?"}, outputs)
    assert res["score"] == 0.0


@pytest.mark.unit
def test_agentevals_trajectory_strict_match():
    """Demonstrates Pattern 1 fix: compare tool trajectory instead of raw AIMessage UUID dicts."""
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    strict_eval = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    actual_msgs = [
        HumanMessage(content="What is the weather in Tokyo?"),
        AIMessage(content="", id="run-dynamic-uuid-1", tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "Tokyo"}}]),
        ToolMessage(content='{"status": "success", "report": "12.1°C"}', name="get_weather", tool_call_id="c1"),
        AIMessage(content="Tokyo is currently 12.1°C.", id="run-dynamic-uuid-2"),
    ]
    ref_msgs = [
        HumanMessage(content="What is the weather in Tokyo?"),
        AIMessage(content="", id="run-golden-uuid-9", tool_calls=[{"id": "c9", "name": "get_weather", "args": {"city": "Tokyo"}}]),
        ToolMessage(content='{"status": "success", "report": "18°C"}', name="get_weather", tool_call_id="c9"),
        AIMessage(content="Tokyo is 18°C.", id="run-golden-uuid-8"),
    ]
    res = strict_eval(outputs=actual_msgs, reference_outputs=ref_msgs)
    assert res["score"] is True


@pytest.mark.unit
def test_extract_tool_context_and_persona_prompt_builder():
    """Verifies OpenEvals {context} extraction and ADK scenarios.json -> simulator prompt mapping."""
    from langchain_core.messages import AIMessage, ToolMessage

    outputs = {
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "1", "name": "get_weather", "args": {"city": "Paris"}}]),
            ToolMessage(content='{"status": "success", "report": "15°C and sunny"}', name="get_weather", tool_call_id="1"),
            AIMessage(content="It is 15°C and sunny in Paris."),
        ]
    }
    ctx = extract_tool_context(outputs)
    assert "get_weather" in ctx
    assert "15°C and sunny" in ctx
    assert extract_final_answer(outputs) == "It is 15°C and sunny in Paris."

    scenarios = load_scenarios()
    assert len(scenarios) >= 3
    prompt = build_persona_simulator_prompt(scenarios[0])
    assert "patient" in prompt
    assert scenarios[0]["starting_prompt"] in prompt


# ==============================================================================
# Gate 2: LangSmith Dataset Seeding & `evaluate()` Experiment Runner
# ==============================================================================
def ensure_langsmith_dataset(
    client: Any = None,
    dataset_name: str = GOLDEN_DATASET_NAME,
) -> str:
    """Creates and seeds the LangSmith dataset if it does not yet exist."""
    from langsmith import Client

    client = client or Client()
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Day 12 golden evaluation cases for city_time_weather_agent (LangGraph parity).",
        )
        client.create_examples(dataset_id=dataset.id, examples=GOLDEN_EXAMPLES)
    return dataset_name


def run_langsmith_experiment(
    dataset_name: str = GOLDEN_DATASET_NAME,
    experiment_prefix: str = "city-weather-ci-gate",
) -> Any:
    """Seeds golden dataset and runs LangSmith evaluate() with Agentevals + OpenEvals + Guardrail."""
    import dotenv
    from langsmith import evaluate
    from city_time_weather_agent.agent_langgraph import get_graph

    dotenv.load_dotenv()
    ensure_langsmith_dataset(dataset_name=dataset_name)
    graph = get_graph()
    evaluators = build_langsmith_evaluators()

    return evaluate(
        lambda inputs: graph.invoke({"messages": [("user", inputs["question"])]}),
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
    )


# ==============================================================================
# Gate 2: Pytest-Native `@pytest.mark.langsmith` Live & Persona Simulations
# ==============================================================================
def _require_langsmith_credentials() -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        pytest.skip("LANGSMITH_API_KEY not set (Pattern 2 CI check).")


@pytest.mark.integration
@pytest.mark.llm_eval
@pytest.mark.langsmith
def test_langgraph_live_smoke():
    """Runs live LangGraph smoke check + custom/hallucination checks logged to LangSmith."""
    _require_langsmith_credentials()
    from langsmith import testing as t
    from city_time_weather_agent.agent_langgraph import get_graph

    question = "What is the weather in Tokyo?"
    t.log_inputs({"question": question})

    graph = get_graph()
    result = graph.invoke({"messages": [("user", question)]})
    answer = extract_final_answer(result)
    tools_called = extract_tool_trajectory(result["messages"])
    t.log_outputs({"answer": answer, "tools_called": tools_called})

    assert "get_weather" in tools_called
    guardrail = unsupported_city_guardrail_ls({"question": question}, result)
    t.log_feedback(key=guardrail["key"], score=guardrail["score"])
    assert guardrail["score"] == 1.0


@pytest.mark.integration
@pytest.mark.llm_eval
@pytest.mark.langsmith
@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda s: s["user_persona"]["id"])
def test_langsmith_persona_simulation(scenario: dict[str, Any]):
    """Simulates Day 7 multi-turn personas (`scenarios.json`) using OpenEvals Simulators."""
    _require_langsmith_credentials()
    from langgraph.checkpoint.memory import InMemorySaver
    from langsmith import testing as t
    from openevals.llm import create_llm_as_judge
    from openevals.simulators import create_llm_simulated_user, run_multiturn_simulation
    from city_time_weather_agent.agent_langgraph import (
        build_gemini_chat_model,
        build_langgraph_agent,
    )

    persona_id = scenario["user_persona"]["id"]
    thread_id = f"sim-{persona_id}"
    checkpointer = InMemorySaver()
    graph = build_langgraph_agent(checkpointer=checkpointer)
    judge_llm = build_gemini_chat_model()

    t.log_inputs({"persona": persona_id, "plan": scenario["conversation_plan"]})

    def app(inputs: dict[str, Any], *, thread_id: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.invoke(
            {"messages": [{"role": "user", "content": inputs["content"]}]},
            config=config,
        )
        return {"role": "assistant", "content": extract_final_answer(state)}

    sim_user = create_llm_simulated_user(
        system=build_persona_simulator_prompt(scenario),
        client=judge_llm,
        fixed_responses=[scenario["starting_prompt"]],
    )
    conversation_judge = create_llm_as_judge(
        prompt=PERSONA_CONVERSATION_PROMPT,
        judge=judge_llm,
        feedback_key="persona_conversation_quality",
    )

    sim_result = run_multiturn_simulation(
        app=app,
        user=sim_user,
        max_turns=3,
        trajectory_evaluators=[conversation_judge],
        thread_id=thread_id,
    )

    full_graph_messages = graph.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", [])
    guardrail = unsupported_city_guardrail_ls(
        {"persona": persona_id},
        {"messages": full_graph_messages},
    )

    t.log_outputs({"trajectory": sim_result["trajectory"]})
    t.log_feedback(key=guardrail["key"], score=guardrail["score"])
    assert guardrail["score"] == 1.0

    for eval_res in sim_result.get("evaluator_results", []):
        t.log_feedback(
            key=eval_res["key"],
            score=eval_res["score"],
            comment=eval_res.get("comment"),
        )
        assert eval_res["score"], f"Persona {persona_id} failed {eval_res}"


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Run Day 12 LangSmith evaluation pipeline.")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only create/seed the LangSmith golden dataset without running evaluate().",
    )
    args = parser.parse_args()
    if args.seed_only:
        name = ensure_langsmith_dataset()
        print(f"Dataset ready in LangSmith: {name}")
    else:
        results = run_langsmith_experiment()
        print("Completed LangSmith evaluate() experiment:", results)
