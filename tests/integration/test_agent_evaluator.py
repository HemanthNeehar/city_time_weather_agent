"""
tests/integration/test_agent_evaluator.py

Layer 1B: ADK AgentEvaluator integration runner.
Evaluates agent trajectory and responses against the golden dataset (evalset1.evalset.json).
"""
from pathlib import Path
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.llm_eval
async def test_city_time_weather_agent_golden_evalset():
    """Runs ADK AgentEvaluator against canonical golden dataset evalset1.evalset.json.
    
    Verifies full multi-turn trajectory (tool calls & arguments) and grounded final responses.
    Gate 2: Runs on PR merge to main and staging release gates.
    """
    eval_dataset = Path(__file__).resolve().parent.parent.parent / "evalset1.evalset.json"
    assert eval_dataset.exists(), f"Golden dataset not found at {eval_dataset}"

    await AgentEvaluator.evaluate(
        agent_module="city_time_weather_agent.agent",
        eval_dataset_file_path_or_dir=str(eval_dataset),
        num_runs=1,
        print_detailed_results=True,
    )

