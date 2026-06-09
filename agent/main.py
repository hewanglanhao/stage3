from __future__ import annotations

import os
import shutil
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
from token_analysis import summarize_token_content


def clean_candidate_dir(log: AgentLog) -> None:
    if os.getenv("AGENT_CLEAN_CANDIDATES", "1") == "0":
        log.log("generation", "candidate directory cleanup disabled", {"path": CANDIDATE_DIR})
        return
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in CANDIDATE_DIR.iterdir():
        if path.is_file() and path.name.startswith("engine_iter_") and path.suffix == ".py":
            path.unlink()
            removed += 1
        elif path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path)
            removed += 1
    log.log("generation", "cleaned previous candidate artifacts", {
        "path": CANDIDATE_DIR,
        "removed_entries": removed,
    })


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    log = AgentLog()
    started = time.perf_counter()
    log.log("agent", "starting automated runtime generation", {"root": ROOT, "workspace": WORKSPACE})
    clean_candidate_dir(log)
    load_env_file(log)

    model_config_path, weight_dir, model_config = resolve_inputs(log)
    env_summary = probe_environment(model_config, weight_dir, log)
    trace_summary = analyze_trace(log)
    trace_summary["token_content_profile"] = summarize_token_content(model_config, log)
    spec = extract_runtime_spec()
    log.log("spec", "extracted runtime hard constraints", spec)

    device = os.getenv("AGENT_DEVICE", "auto")
    deterministic_candidates = generate_deterministic_candidates(log)
    results: list[CandidateResult] = []
    max_candidates = int(os.getenv("AGENT_MAX_CANDIDATES", "6"))
    llm = LLMClient(log)
    max_consecutive_llm_failures = max(1, int(os.getenv("AGENT_MAX_CONSECUTIVE_LLM_FAILURES", "3")))
    consecutive_llm_failures = 0

    for candidate in deterministic_candidates[:max_candidates]:
        evaluate_candidate(candidate, model_config_path, weight_dir, model_config, device, log)
        results.append(candidate)

    if len(results) < max_candidates and os.getenv("AGENT_ENABLE_TOKEN_AWARE_BRANCH", "1") != "0":
        log.log("llm", "starting token-aware LLM optimization branch", {
            "evaluated_candidates": len(results),
            "max_candidates": max_candidates,
            "raw_token_values_redacted": True,
        })
        token_candidate = maybe_generate_llm_candidate(
            llm,
            env_summary,
            trace_summary,
            spec,
            results,
            log,
            llm_round=0,
            branch_mode="token_aware",
        )
        if token_candidate is not None:
            evaluate_candidate(token_candidate, model_config_path, weight_dir, model_config, device, log)
            results.append(token_candidate)
            consecutive_llm_failures = 0
        else:
            consecutive_llm_failures += 1
            log.log("llm", "token-aware LLM branch did not produce a usable candidate", {
                "consecutive_failures": consecutive_llm_failures,
                "failure_limit": max_consecutive_llm_failures,
            })

    llm_round = 1
    while len(results) < max_candidates:
        log.log("llm", f"starting LLM optimization round {llm_round}", {
            "evaluated_candidates": len(results),
            "max_candidates": max_candidates,
        })
        llm_candidate = maybe_generate_llm_candidate(
            llm,
            env_summary,
            trace_summary,
            spec,
            results,
            log,
            llm_round,
        )
        if llm_candidate is None:
            consecutive_llm_failures += 1
            log.log("llm", f"LLM round {llm_round} did not produce a usable candidate", {
                "consecutive_failures": consecutive_llm_failures,
                "failure_limit": max_consecutive_llm_failures,
            })
            llm_round += 1
            if consecutive_llm_failures >= max_consecutive_llm_failures:
                log.log("llm", "stopping LLM loop after consecutive failed rounds", {
                    "consecutive_failures": consecutive_llm_failures,
                    "failure_limit": max_consecutive_llm_failures,
                })
                break
            continue
        consecutive_llm_failures = 0
        evaluate_candidate(llm_candidate, model_config_path, weight_dir, model_config, device, log)
        results.append(llm_candidate)
        llm_round += 1

    best = pick_best(results, log)
    install_best(best, log)
    write_output_report(env_summary, trace_summary, spec, results, best, log, llm=llm)
    elapsed = time.perf_counter() - started
    log.log("agent", "finished automated runtime generation", {
        "elapsed_s": round(elapsed, 3),
        "final_engine": WORKSPACE / "engine.py",
        "output_report": WORKSPACE / "output3.md",
        "best_candidate": best.name,
    })


if __name__ == "__main__":
    main()
