from __future__ import annotations

import os
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from candidate_manager import generate_deterministic_candidates, maybe_generate_llm_candidate
from common import AgentLog, CANDIDATE_DIR, WORKSPACE, CandidateResult, ROOT
from env_loader import load_env_file
from llm_client import LLMClient
from local_evaluator import evaluate_candidate
from probe import probe_environment, resolve_inputs
from report import write_output_report
from runtime_spec import extract_runtime_spec
from selection import install_best, pick_best
from trace_analysis import analyze_trace


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    log = AgentLog()
    started = time.perf_counter()
    log.log("agent", "starting automated runtime generation", {"root": ROOT, "workspace": WORKSPACE})
    load_env_file(log)

    model_config_path, weight_dir, model_config = resolve_inputs(log)
    env_summary = probe_environment(model_config, weight_dir, log)
    trace_summary = analyze_trace(log)
    spec = extract_runtime_spec()
    log.log("spec", "extracted runtime hard constraints", spec)

    device = os.getenv("AGENT_DEVICE", "auto")
    deterministic_candidates = generate_deterministic_candidates(log)
    results: list[CandidateResult] = []
    max_candidates = int(os.getenv("AGENT_MAX_CANDIDATES", "4"))
    llm = LLMClient(log)

    for candidate in deterministic_candidates[:max_candidates]:
        evaluate_candidate(candidate, model_config_path, weight_dir, model_config, device, log)
        results.append(candidate)

    if len(results) < max_candidates:
        llm_candidate = maybe_generate_llm_candidate(llm, env_summary, trace_summary, spec, results, log)
        if llm_candidate is not None:
            evaluate_candidate(llm_candidate, model_config_path, weight_dir, model_config, device, log)
            results.append(llm_candidate)

    best = pick_best(results, log)
    install_best(best, log)
    write_output_report(env_summary, trace_summary, spec, results, best, log)
    elapsed = time.perf_counter() - started
    log.log("agent", "finished automated runtime generation", {
        "elapsed_s": round(elapsed, 3),
        "final_engine": WORKSPACE / "engine.py",
        "output_report": WORKSPACE / "output3.md",
        "best_candidate": best.name,
    })


if __name__ == "__main__":
    main()
