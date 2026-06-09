from __future__ import annotations

import json
import os
import textwrap
from typing import Any

from common import AgentLog, CandidateResult, to_jsonable


PREFERRED_AGGRESSIVE_GUIDANCE = [
    "When correctness is passing, prefer fused QKV projection at Engine init: concatenate q/k/v weights once, run one F.linear per layer, split q/k/v, and avoid keeping duplicate GPU copies of unfused q/k/v weights.",
    "Fuse MLP gate/up projection weights similarly: one F.linear for gate_up, split the result, apply SiLU(gate) * up, then down projection.",
    "Use shared packed KV cache blocks for requests produced by the same batched prefill/decode group. Track length, shared cache block, and row index per request; gather rows for decode and repack after decode.",
    "Keep the old generic per-request KV concat path as a correctness fallback for heterogeneous cache blocks, prefill replacement, remove, and mixed-length states.",
]


def build_feedback(
    results: list[CandidateResult],
    trace_summary: dict[str, Any],
    env_summary: dict[str, Any] | None = None,
    spec: list[str] | None = None,
    llm: Any | None = None,
    log: AgentLog | None = None,
) -> dict[str, Any]:
    local_feedback = build_local_feedback(results, trace_summary)
    if not should_use_llm_feedback(llm):
        return local_feedback

    prompt = build_feedback_prompt(
        local_feedback=local_feedback,
        trace_summary=trace_summary,
        env_summary=env_summary or {},
        spec=spec or [],
    )
    llm_feedback = llm.ask_for_feedback(prompt)
    if not llm_feedback:
        if log is not None:
            log.log("feedback", "LLM feedback summary unavailable; using local feedback")
        return local_feedback

    merged = merge_llm_feedback(local_feedback, llm_feedback)
    if log is not None:
        log.log("feedback", "LLM summarized current defects and guidance", {
            "priority": merged.get("priority", ""),
            "defect_count": len(merged.get("defects", [])),
            "guidance_count": len(merged.get("guidance", [])),
            "details_omitted": True,
        })
    return merged


def should_use_llm_feedback(llm: Any | None) -> bool:
    if llm is None or not getattr(llm, "enabled", False):
        return False
    return os.getenv("AGENT_USE_LLM_FEEDBACK", "1") != "0"


def build_feedback_prompt(
    local_feedback: dict[str, Any],
    trace_summary: dict[str, Any],
    env_summary: dict[str, Any],
    spec: list[str],
) -> str:
    return textwrap.dedent(f"""
    You are summarizing the current defects of an automatically generated LLM inference runtime.
    The local evaluator has already run correctness/stress/benchmark. Your job is not to write code.
    Your job is to produce concise, actionable defects and guidance for the next engine-generation prompt.

    Hard priorities:
    1. If correctness or request-state stress failed, guidance must focus only on fixing correctness.
    2. If correctness passed, compare benchmark rows and identify the main throughput bottleneck.
    3. Guidance must preserve create_engine/prefill/decode/remove and dynamic config loading.
    4. Do not recommend hard-coding model dimensions or request ids.

    Preferred aggressive guidance when correctness is passing:
    {json.dumps(PREFERRED_AGGRESSIVE_GUIDANCE, indent=2, ensure_ascii=False)}

    Runtime spec:
    {json.dumps(to_jsonable(spec), indent=2, ensure_ascii=False)[:2500]}

    Environment/model summary:
    {json.dumps(to_jsonable(env_summary), indent=2, ensure_ascii=False)[:3500]}

    Trace summary:
    {json.dumps(to_jsonable(trace_summary), indent=2, ensure_ascii=False)[:3500]}

    Local raw feedback and candidate table:
    {json.dumps(to_jsonable(local_feedback), indent=2, ensure_ascii=False)[:7000]}

    Return strict JSON only with this schema:
    {{
      "summary": "one sentence summary of current state",
      "priority": "correctness" | "performance" | "fallback",
      "defects": ["short defect 1", "short defect 2"],
      "guidance": ["actionable instruction 1", "actionable instruction 2"],
      "risk_notes": ["risk to avoid 1", "risk to avoid 2"]
    }}
    """).strip()


def merge_llm_feedback(local_feedback: dict[str, Any], llm_feedback: dict[str, Any]) -> dict[str, Any]:
    defects = normalize_string_list(llm_feedback.get("defects")) or local_feedback.get("defects", [])
    guidance = normalize_string_list(llm_feedback.get("guidance")) or local_feedback.get("guidance", [])
    risk_notes = normalize_string_list(llm_feedback.get("risk_notes"))
    return {
        "source": "llm_feedback",
        "summary": str(llm_feedback.get("summary", "")).strip(),
        "priority": str(llm_feedback.get("priority", "")).strip() or infer_priority(local_feedback),
        "defects": defects[:8],
        "guidance": guidance[:10],
        "risk_notes": risk_notes[:6],
        "local_defects": local_feedback.get("defects", []),
        "local_guidance": local_feedback.get("guidance", []),
        "current_best": local_feedback.get("current_best"),
        "latest_candidate": local_feedback.get("latest_candidate"),
        "candidate_table": local_feedback.get("candidate_table", []),
    }


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def infer_priority(local_feedback: dict[str, Any]) -> str:
    latest = local_feedback.get("latest_candidate") or {}
    if not latest.get("correctness_ok") or not latest.get("stress_ok"):
        return "correctness"
    if latest.get("benchmark_ok"):
        return "performance"
    return "fallback"


def build_local_feedback(results: list[CandidateResult], trace_summary: dict[str, Any]) -> dict[str, Any]:
    passing = [result for result in results if result.correctness_ok and result.stress_ok]
    best = max(passing, key=lambda result: (result.benchmark_ok, result.score)) if passing else None
    latest = results[-1] if results else None
    defects: list[str] = []
    guidance: list[str] = []

    if latest is None:
        defects.append("No candidate has been evaluated yet.")
        guidance.append("Start from a conservative engine that matches the public interface exactly.")
    elif not latest.precheck_ok:
        defects.append(f"Latest candidate `{latest.name}` failed import/syntax precheck: {latest.failure_reason[-700:]}")
        guidance.append("Return a complete syntactically valid engine.py with create_engine and Engine methods.")
    elif not latest.correctness_ok:
        defects.append(f"Latest candidate `{latest.name}` failed official correctness: {latest.failure_reason[-900:]}")
        guidance.append("Fix correctness before performance. Preserve request state, request_ids row order, RoPE positions, dtype behavior, and prefill replacement semantics.")
    elif not latest.stress_ok:
        defects.append(f"Latest candidate `{latest.name}` failed custom request-state stress tests: {latest.failure_reason[-900:]}")
        guidance.append("Focus on prefill replacement, non-contiguous request ids, remove semantics, and mixed-length active requests.")
    elif not latest.benchmark_ok:
        defects.append(f"Latest candidate `{latest.name}` passed correctness but benchmark data is unavailable: {latest.failure_reason[-700:]}")
        guidance.append("Keep the correct structure and reduce runtime overhead without adding fragile dependencies.")
    else:
        if best is not None and latest is not best and latest.score <= best.score:
            defects.append(
                f"Latest candidate `{latest.name}` did not beat current best `{best.name}`. "
                f"latest_score={latest.score:.3f}, best_score={best.score:.3f}."
            )
        defects.extend(compare_with_best(latest, best))
        bottleneck = infer_bottleneck(latest)
        if bottleneck:
            defects.append(bottleneck)

        guidance.append("Do not regress correctness; keep create_engine/prefill/decode/remove signatures unchanged.")
        guidance.append("Prefer aggressive but locally checkable runtime changes that improve mixed and decode throughput, especially fused QKV projection and fused gate/up projection built once at Engine init.")
        guidance.append("For decode-heavy or mixed traces, try shared packed KV cache blocks for requests from the same batch; track per-request row indices and keep a generic fallback for heterogeneous states.")
        guidance.append("Use config-derived dimensions only; never hard-code hidden model parameters.")
        if trace_summary.get("max_decode_step_seen", 0):
            guidance.append("Observed traces include repeated decode after prefill, so KV cache and batched equal-length decode are high-value paths.")
        if trace_summary.get("prompt_lengths"):
            guidance.append(f"Observed prompt lengths include {trace_summary.get('prompt_lengths')}; preserve grouped prefill by prompt length.")

    if best is None:
        guidance.append("A safe full-recompute fallback already exists; any LLM patch must pass official correctness before selection.")
    else:
        guidance.append(f"Current best is `{best.name}` with score={best.score:.3f}; propose a full engine.py that can beat it or fix the latest failure.")

    return {
        "source": "local_judge",
        "current_best": summarize_result(best) if best else None,
        "latest_candidate": summarize_result(latest) if latest else None,
        "defects": defects[:8],
        "guidance": guidance[:10],
        "candidate_table": [summarize_result(result) for result in results],
    }


def summarize_result(result: CandidateResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "iteration": result.iteration,
        "name": result.name,
        "precheck_ok": result.precheck_ok,
        "correctness_ok": result.correctness_ok,
        "stress_ok": result.stress_ok,
        "benchmark_ok": result.benchmark_ok,
        "score": result.score,
        "benchmark_summary": summarize_benchmark(result.benchmark),
        "failure_reason_tail": result.failure_reason[-900:],
        "strategy": result.strategy,
        "risk": result.risk,
    }


def summarize_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        case = str(row.get("case_name"))
        summary[case] = {
            "tokens_per_second": row.get("tokens_per_second"),
            "decode_tokens_per_second": row.get("decode_tokens_per_second"),
            "elapsed_ms": row.get("elapsed_ms"),
            "peak_memory_mb": row.get("peak_memory_mb"),
        }
    return summary


def compare_with_best(latest: CandidateResult, best: CandidateResult | None) -> list[str]:
    if best is None or latest is best or not latest.benchmark or not best.benchmark:
        return []
    defects: list[str] = []
    latest_by_case = {str(row.get("case_name")): row for row in latest.benchmark}
    best_by_case = {str(row.get("case_name")): row for row in best.benchmark}
    for case_name in sorted(set(latest_by_case) & set(best_by_case)):
        latest_row = latest_by_case.get(case_name, {})
        best_row = best_by_case.get(case_name, {})
        latest_tps = float(latest_row.get("tokens_per_second", 0.0) or 0.0)
        best_tps = float(best_row.get("tokens_per_second", 0.0) or 0.0)
        if best_tps and latest_tps < best_tps * 0.98:
            defects.append(f"{case_name} throughput regressed versus best: latest={latest_tps:.2f} tok/s, best={best_tps:.2f} tok/s.")
    return defects


def infer_bottleneck(result: CandidateResult) -> str | None:
    if not result.benchmark:
        return None
    by_case = {str(row.get("case_name")): row for row in result.benchmark}

    def total_tps(*case_names: str) -> float:
        for case_name in case_names:
            value = float(by_case.get(case_name, {}).get("tokens_per_second", 0.0) or 0.0)
            if value:
                return value
        return 0.0

    def decode_tps(*case_names: str) -> float:
        for case_name in case_names:
            value = float(by_case.get(case_name, {}).get("decode_tokens_per_second", 0.0) or 0.0)
            if value:
                return value
        return 0.0

    prefill_tps = total_tps("case1_sessions_1_4_prefill_4x128", "prefill")
    decode_rate = decode_tps("case2_sessions_5_8_decode_8x128x16", "decode")
    mixed_short = total_tps("case3_sessions_9_12_mixed_64_128_32", "mixed")
    mixed_long = total_tps("case4_sessions_13_16_mixed_128_all", "mixed")

    if decode_rate and prefill_tps and decode_rate < prefill_tps * 0.4:
        return f"Decode remains the main bottleneck: decode={decode_rate:.2f} decode tok/s versus prefill={prefill_tps:.2f} tok/s."
    if mixed_short and prefill_tps and mixed_short < prefill_tps:
        return f"Mixed short-insert throughput is below prefill-only throughput: mixed={mixed_short:.2f}, prefill={prefill_tps:.2f}."
    if mixed_long and prefill_tps and mixed_long < prefill_tps:
        return f"Mixed all-128 throughput is below prefill-only throughput: mixed={mixed_long:.2f}, prefill={prefill_tps:.2f}."
    return None
