#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
"${PYTHON:-python3}" stage3/agent.py > results.log 2>&1
