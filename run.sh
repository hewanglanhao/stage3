#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="$SCRIPT_DIR/stage3/doc/环境变量.txt"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing LLM environment file: $ENV_FILE" >&2
    exit 1
fi

# Load this file explicitly so its model/API settings override any values
# inherited from the shell that launches the agent.
# shellcheck disable=SC1090
source <(sed 's/\r$//' "$ENV_FILE")
export AGENT_ENV_FILE="$ENV_FILE"

: "${API_KEY:?API_KEY is required in $ENV_FILE}"
: "${BASE_MODEL:?BASE_MODEL is required in $ENV_FILE}"
: "${BASE_URL:?BASE_URL is required in $ENV_FILE}"

# Default to using the external LLM optimization round.
# API_KEY / BASE_URL / BASE_MODEL were loaded from ENV_FILE above.
export AGENT_USE_LLM="${AGENT_USE_LLM:-1}"
export AGENT_MAX_CANDIDATES="${AGENT_MAX_CANDIDATES:-9}"
export AGENT_CLEAN_CANDIDATES="${AGENT_CLEAN_CANDIDATES:-1}"

# Try one token-profile-aware LLM branch after the deterministic baseline.
# The prompt uses only redacted aggregate token statistics, not raw token ids.
export AGENT_ENABLE_TOKEN_AWARE_BRANCH="${AGENT_ENABLE_TOKEN_AWARE_BRANCH:-1}"
export AGENT_TOKEN_PROFILE="${AGENT_TOKEN_PROFILE:-1}"

# LLM API calls can be slow for large feedback/engine-generation prompts.
export LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-120}"
export LLM_FEEDBACK_TIMEOUT_SECONDS="${LLM_FEEDBACK_TIMEOUT_SECONDS:-120}"
export LLM_CANDIDATE_TIMEOUT_SECONDS="${LLM_CANDIDATE_TIMEOUT_SECONDS:-300}"
export LLM_RETRIES="${LLM_RETRIES:-1}"

# Benchmark controls. The four hard-coded real trace cases always run;
# only the number of repeats is configurable here.
# Example:
#   AGENT_BENCH_REPEAT=1 bash stage3/run.sh
export AGENT_BENCH_REPEAT="${AGENT_BENCH_REPEAT:-2}"
export AGENT_BENCH_WARMUP="${AGENT_BENCH_WARMUP:-0}"

"${PYTHON:-python3}" stage3/agent.py > results.log 2>&1
