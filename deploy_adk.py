"""
deploy_adk.py

Day 9: Deploy city_time_weather_agent to Vertex AI Agent Engine with
OpenTelemetry prompt/response capture and Eventarc CI/CD Gate 3 safety.
"""

import argparse
import os
from typing import Dict, Optional

import dotenv
import vertexai
from vertexai import agent_engines

from city_time_weather_agent.agent import root_agent

# Keys truly reserved by Cloud Run / Agent Engine (PORT, K_SERVICE, etc.).
# Note: GOOGLE_CLOUD_LOCATION SHOULD be passed as "global" for preview Gemini models.
RESERVED_RUNTIME_ENV_KEYS = frozenset(
    {
        "PORT",
        "K_SERVICE",
        "K_REVISION",
        "K_CONFIGURATION",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
    }
)

# Universal 4-variable OpenTelemetry capture + global Vertex AI model routing config
OTEL_CAPTURE_ENV_VARS: Dict[str, str] = {
    "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
    "GOOGLE_CLOUD_LOCATION": "global",
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
    "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
    "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS": "true",
}


def build_agent_env_vars(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Builds the runtime env_vars dict while stripping reserved GCP runtime keys."""
    merged = dict(OTEL_CAPTURE_ENV_VARS)
    if extra_env:
        merged.update(extra_env)
    return {k: v for k, v in merged.items() if k not in RESERVED_RUNTIME_ENV_KEYS}


def deploy(
    project: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    display_name: str = "City Time & Weather Agent",
    dry_run: bool = False,
):
    """Wraps root_agent in AdkApp and deploys to Vertex AI Agent Engine."""
    dotenv.load_dotenv()

    project_id = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = location or os.environ.get("AGENT_ENGINE_LOCATION", "us-central1")
    bucket = staging_bucket or os.environ.get("GOOGLE_CLOUD_STAGING_BUCKET")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT must be set in environment or passed via --project.")

    env_vars = build_agent_env_vars()

    if dry_run:
        return {
            "project": project_id,
            "location": region,
            "staging_bucket": bucket,
            "display_name": display_name,
            "agent_name": root_agent.name,
            "env_vars": env_vars,
        }

    app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)

    vertexai.init(project=project_id, location=region, staging_bucket=bucket)
    remote_agent = agent_engines.create(
        agent_engine=app,
        requirements=[
            "google-adk>=2.13.0",
            "google-cloud-aiplatform[adk,agent_engines]",
            "google-genai>=2.4.0"
        ],
        extra_packages=["city_time_weather_agent"],
        display_name=display_name,
        description="Dual-tool ADK agent (weather & local time) with OTel payload capture.",
        env_vars=env_vars,
    )
    print(f"Deployed ReasoningEngine resource: {remote_agent.resource_name}")
    return remote_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy ADK agent to Vertex AI Agent Engine.")
    parser.add_argument("--project", help="GCP Project ID (defaults to GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--location", default="us-central1", help="Agent Engine region (default: us-central1)")
    parser.add_argument("--staging-bucket", help="GCS staging bucket gs://... (defaults to GOOGLE_CLOUD_STAGING_BUCKET)")
    parser.add_argument("--display-name", default="City Time & Weather Agent")
    parser.add_argument("--dry-run", action="store_true", help="Validate payload locally without deploying")
    args = parser.parse_args()

    result = deploy(
        project=args.project,
        location=args.location,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Dry-run deployment payload verified:")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
