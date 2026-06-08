from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOURCE_ROOT.parent
WORKSPACE = ROOT
CANDIDATE_DIR = WORKSPACE / "candidates"
EVALUATOR_DIR = SOURCE_ROOT / "evaluator"


@dataclass
class CandidateResult:
    iteration: int
    name: str
    strategy: str
    expected_benefit: str
    risk: str
    path: Path
    precheck_ok: bool = False
    correctness_ok: bool = False
    stress_ok: bool = False
    benchmark_ok: bool = False
    correctness_output: str = ""
    stress_output: str = ""
    benchmark: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""
    score: float = float("-inf")
    llm_notes: str = ""


class AgentLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log(self, module: str, message: str, data: Any | None = None) -> None:
        event = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "module": module,
            "message": message,
            "data": data,
        }
        self.events.append(event)
        print(f"[{event['time']}] [{module}] {message}", flush=True)
        if data is not None:
            print(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), flush=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return to_jsonable(value.__dict__)
    return value


def first_existing_path(*paths: str | Path) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def run_subprocess(cmd: list[str], timeout: int, cwd: Path = ROOT) -> tuple[int, str]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        output = proc.stdout or ""
        return proc.returncode, output + f"\n[agent_subprocess_elapsed_s={elapsed:.3f}]"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return 124, output + f"\n[agent_timeout_s={timeout}]"


def resolve_device(device: str) -> str:
    try:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        if device == "auto" or device.startswith("cuda"):
            return "cpu"
    return device
