import inspect
import pytest
from city_time_weather_agent.agent import root_agent, get_weather, get_current_time

@pytest.mark.unit
class TestAgentContracts:

    def test_agent_tool_binding(self):
        """Ensure all required tools are attached to root_agent."""
        tool_names = [t.__name__ for t in root_agent.tools]
        assert "get_weather" in tool_names
        assert "get_current_time" in tool_names

    def test_tool_docstrings_exist_for_schema_generation(self):
        """Crucial ADK Requirement: Tools without docstrings generate empty OpenAPI descriptions, causing Gemini to misroute calls."""
        for tool in [get_weather, get_current_time]:
            doc = inspect.getdoc(tool)
            assert doc is not None and len(doc.strip()) > 10, f"Tool {tool.__name__} has missing or weak docstring!"

    def test_tool_parameter_type_hints(self):
        """Tools must have strict type hints for ADK schema serialization."""
        for tool in [get_weather, get_current_time]:
            sig = inspect.signature(tool)
            assert "city" in sig.parameters
            assert sig.parameters["city"].annotation is str, f"Tool {tool.__name__} missing type hint on 'city'"
            assert sig.return_annotation is dict, f"Tool {tool.__name__} must declare return type dict"

    def test_instruction_explicitly_forbids_guessing(self):
        """Guardrail rule: Instruction must enforce tool-first verification."""
        instruction = root_agent.instruction.lower()
        assert "never guess" in instruction or "always call a tool" in instruction