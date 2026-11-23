#!/usr/bin/env bash
# scripts/setup_online_monitoring.sh
# Day 10 (Gate 4): Deploy Cloud Monitoring Alert Policy for Vertex AI Online Evals

set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
POLICY_FILE="${POLICY_FILE:-./policy.yaml}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: GOOGLE_CLOUD_PROJECT is not set."
  exit 1
fi

echo "Applying Gate 4 Cloud Monitoring Drift Alert Policy to project: ${PROJECT_ID}"
gcloud monitoring policies create \
  --project="${PROJECT_ID}" \
  --policy-from-file="${POLICY_FILE}"

echo "Gate 4 alert policy created. Remember to attach notification channels in Cloud Monitoring."
