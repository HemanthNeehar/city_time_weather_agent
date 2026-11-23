"""
agent_langgraph.py

Day 12: Cross-Framework Parity — Rebuilding `city_time_weather_agent` on
LangChain + LangGraph using the exact same weather & local-time tools as `agent.py`.
"""

import os
from typing import Any

import dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from city_time_weather_agent.agent import get_current_time, get_weather

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the `get_weather` tool for weather questions and the "
    "`get_current_time` tool for time questions. Always call a tool before answering; never guess."
)


def build_gemini_chat_model(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """Builds a Gemini chat model supporting either Vertex AI or Gemini API key auth."""
    dotenv.load_dotenv()

    # Supports either Vertex AI (GOOGLE_GENAI_USE_VERTEXAI=TRUE) or Gemini API Key.
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        vertexai=use_vertex if use_vertex else None,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT") if use_vertex else None,
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global") if use_vertex else None,
    )


def build_langgraph_agent(
    model: str = "gemini-2.5-flash",
    checkpointer: Any = None,
) -> Any:
    """Builds and returns the compiled LangGraph agent graph (via LangChain v1 create_agent)."""
    llm = build_gemini_chat_model(model=model)
    return create_agent(
        model=llm,
        tools=[get_weather, get_current_time],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def get_graph() -> Any:
    """Lazy accessor so importing this module in unit tests does not require live LLM keys."""
    return build_langgraph_agent()
