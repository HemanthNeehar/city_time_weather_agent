#!/usr/bin/env bash
# scripts/run_aqua_eval.sh
# Day 11 (Gate 5): Ambient Quality Agent (AQuA) Outer-to-Inner Loop Runner
# Closes the loop from production BigQuery _AllSpans -> Verified Insights -> Code RCA.

set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
AGENT_NAME="${AGENT_NAME:-city_time_weather_agent}"
AGENT_SOURCE_DIR="${AGENT_SOURCE_DIR:-./city_time_weather_agent}"
GOAL_FILE="${GOAL_FILE:-./city_time_weather_agent/goal.md}"
TELEMETRY_DATASET="${TELEMETRY_DATASET:-trace_spans_linked}"
TELEMETRY_TABLE="${TELEMETRY_TABLE:-_AllSpans}"

# Fix Pattern 1: Override upstream recipe default (gemini-3.7-flash -> 404 NOT_FOUND)
export AQA_BASE_MODEL="${AQA_BASE_MODEL:-gemini-3-flash-preview}"
export AQA_VERIFICATION_MODEL="${AQA_VERIFICATION_MODEL:-gemini-3-flash-preview}"
export AQA_INSIGHTS_MODEL="${AQA_INSIGHTS_MODEL:-gemini-3.1-pro-preview}"

MODE="${1:-eval}"

case "${MODE}" in
  --link-traces)
    echo "[1/3] Linking Cloud Trace -> BigQuery ${TELEMETRY_DATASET}.${TELEMETRY_TABLE}..."
    agents-cli aqua link-traces --project="${PROJECT_ID}" --apply
    ;;
  --onboard)
    echo "[2/3] Onboarding ${AGENT_NAME} with goal specification (${GOAL_FILE})..."
    agents-cli aqua onboard \
      --project="${PROJECT_ID}" \
      --agent-name="${AGENT_NAME}" \
      --source-dir="${AGENT_SOURCE_DIR}" \
      --goal-file="${GOAL_FILE}" \
      --telemetry-dataset="${TELEMETRY_DATASET}" \
      --telemetry-table="${TELEMETRY_TABLE}"
    ;;
  --insights)
    echo "Fetching critic-verified production insights for ${AGENT_NAME}..."
    agents-cli aqua insights --agent-name="${AGENT_NAME}"
    ;;
  --rca)
    INSIGHT_ID="${2:?Usage: $0 --rca <INSIGHT_ID>}"
    echo "Running code-anchored RCA for insight ${INSIGHT_ID}..."
    agents-cli aqua rca "${INSIGHT_ID}" --agent-name="${AGENT_NAME}"
    ;;
  *)
    echo "Running nightly AQuA outer-loop investigation on ${TELEMETRY_DATASET}.${TELEMETRY_TABLE}..."
    agents-cli aqua evaluate \
      --project="${PROJECT_ID}" \
      --agent-name="${AGENT_NAME}" \
      --goal-file="${GOAL_FILE}" \
      --telemetry-dataset="${TELEMETRY_DATASET}" \
      --telemetry-table="${TELEMETRY_TABLE}"
    ;;
esac
