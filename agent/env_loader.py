from __future__ import annotations

import os
import shlex
from pathlib import Path

from common import AgentLog, SOURCE_ROOT

DEFAULT_ENV_FILES = [
    SOURCE_ROOT / "doc" / "环境变量.txt",
    SOURCE_ROOT.parent / "stage3" / "doc" / "环境变量.txt",
]


def load_env_file(log: AgentLog) -> dict[str, str]:
    env_file = os.getenv("AGENT_ENV_FILE")
    candidates = [Path(env_file)] if env_file else DEFAULT_ENV_FILES
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        log.log("env", "no agent env file found; relying on process environment")
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_export_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = "loaded"
        else:
            loaded[key] = "already_set"

    log.log("env", "loaded agent environment file", {
        "path": path,
        "keys": sorted(loaded.keys()),
        "secret_values_redacted": True,
    })
    return loaded


def parse_export_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    try:
        parts = shlex.split(value, comments=False, posix=True)
        value = parts[0] if parts else ""
    except ValueError:
        value = value.strip().strip('"').strip("'")
    return key, value
