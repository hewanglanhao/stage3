#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Default to using the external LLM optimization round.
# The agent loads API_KEY / BASE_URL / BASE_MODEL from stage3/doc/环境变量.txt.
export AGENT_USE_LLM="${AGENT_USE_LLM:-1}"
export AGENT_MAX_CANDIDATES="${AGENT_MAX_CANDIDATES:-6}"
export AGENT_CLEAN_CANDIDATES="${AGENT_CLEAN_CANDIDATES:-1}"

# Try one token-profile-aware LLM branch after deterministic candidates.
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
