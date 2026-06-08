#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Default to using the external LLM optimization round.
# The agent loads API_KEY / BASE_URL / BASE_MODEL from stage3/doc/环境变量.txt.
export AGENT_USE_LLM="${AGENT_USE_LLM:-1}"
export AGENT_MAX_CANDIDATES="${AGENT_MAX_CANDIDATES:-6}"

# Benchmark controls. The four hard-coded real trace cases always run;
# only the number of repeats is configurable here.
# Example:
#   AGENT_BENCH_REPEAT=1 bash stage3/run.sh
export AGENT_BENCH_REPEAT="${AGENT_BENCH_REPEAT:-2}"
export AGENT_BENCH_WARMUP="${AGENT_BENCH_WARMUP:-0}"

"${PYTHON:-python3}" stage3/agent.py > results.log 2>&1
