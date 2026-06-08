from __future__ import annotations

from pathlib import Path
from typing import Any

from common import AgentLog, CANDIDATE_DIR, CandidateResult
from llm_client import LLMClient, build_llm_prompt, extract_python_code
from templates import render_kv_cache, render_safe_baseline


def write_candidate(iteration: int, name: str, code: str, strategy: str, benefit: str, risk: str, llm_notes: str = "") -> CandidateResult:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    path = CANDIDATE_DIR / f"engine_iter_{iteration}_{name}.py"
    path.write_text(code, encoding="utf-8")
    return CandidateResult(
        iteration=iteration,
        name=name,
        strategy=strategy,
        expected_benefit=benefit,
        risk=risk,
        path=path,
        llm_notes=llm_notes,
    )


def generate_deterministic_candidates(log: AgentLog) -> list[CandidateResult]:
    candidates = [
        write_candidate(
            1,
            "safe_baseline",
            render_safe_baseline(),
            "Full recompute for every request using the reference math; stores token history only.",
            "Maximum correctness safety and a reliable fallback engine.",
            "Decode throughput is poor because every decode recomputes the whole sequence.",
        ),
        write_candidate(
            2,
            "kv_cache_torch",
            render_kv_cache("kv_cache_torch", group_decode=False, group_prefill=True),
            "Per-request KV cache with grouped prefill; decode computes only the new token but processes requests one by one.",
            "Large decode speedup over full recompute while preserving simple state semantics.",
            "Less efficient when many equal-length requests decode together because Python still loops per request.",
        ),
        write_candidate(
            3,
            "optimized_torch",
            render_kv_cache("optimized_torch", group_decode=True, group_prefill=True),
            "KV cache plus grouped batched prefill/decode, inference_mode, dtype-aware weights, and mask reuse.",
            "Improves official decode and mixed traces where active requests share prompt/decode lengths.",
            "Grouped decode must preserve logits order and correctly split variable-length active requests.",
        ),
    ]
    log.log("generation", "wrote deterministic runtime candidates", [{"name": c.name, "path": c.path} for c in candidates])
    return candidates


def maybe_generate_llm_candidate(
    llm: LLMClient,
    env_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    spec: list[str],
    results: list[CandidateResult],
    log: AgentLog,
) -> CandidateResult | None:
    best_so_far = None
    passing = [r for r in results if r.correctness_ok and r.stress_ok]
    if passing:
        best_so_far = max(passing, key=lambda r: (r.benchmark_ok, r.score))
    elif results:
        best_so_far = results[-1]
    excerpt = ""
    if best_so_far and best_so_far.path.exists():
        excerpt = best_so_far.path.read_text(encoding="utf-8")[:6000]
    prompt = build_llm_prompt(env_summary, trace_summary, spec, results, excerpt)
    parsed = llm.ask_for_candidate(prompt)
    if not parsed:
        return None
    code = extract_python_code(parsed.get("patch_or_full_engine", ""))
    if not code:
        log.log("llm", "LLM proposal did not include a full engine.py; not applying blindly", {
            "strategy": parsed.get("strategy", ""),
            "risk": parsed.get("risk", ""),
        })
        return None
    return write_candidate(
        len(results) + 1,
        "llm_candidate",
        code,
        parsed.get("strategy", "LLM full-engine proposal"),
        parsed.get("expected_benefit", "LLM proposed optimization"),
        parsed.get("risk", "Unknown; validated by local tests before selection."),
        parsed.get("self_check_notes", ""),
    )
