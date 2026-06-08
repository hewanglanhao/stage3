from __future__ import annotations

import json
from typing import Any

from common import AgentLog, CandidateResult, WORKSPACE, to_jsonable
from feedback import build_feedback


def redact_trace_for_report(trace_summary: dict[str, Any]) -> dict[str, Any]:
    if "token_content_profile" not in trace_summary:
        return trace_summary
    redacted = dict(trace_summary)
    profile = trace_summary.get("token_content_profile") or {}
    redacted["token_content_profile"] = {
        "available": bool(profile.get("available")) if isinstance(profile, dict) else False,
        "raw_token_values_redacted": True,
        "used_for_token_aware_llm_branch": True,
        "observations": profile.get("observations", []) if isinstance(profile, dict) else [],
        "full_profile_omitted_from_report": True,
    }
    return redacted


def write_output_report(
    env_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    spec: list[str],
    results: list[CandidateResult],
    best: CandidateResult,
    log: AgentLog,
    llm: object | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Agent Runtime Generation Report")
    lines.append("")
    lines.append("## Final Selection")
    lines.append("- Final engine: `/workspace/engine.py`")
    lines.append(f"- Selected candidate: iter {best.iteration} `{best.name}`")
    lines.append(f"- Correctness: official={best.correctness_ok}, stress={best.stress_ok}")
    lines.append(f"- Benchmark score: {best.score:.3f}")
    lines.append("")
    lines.append("## Environment Probe")
    lines.append("```json")
    lines.append(json.dumps(to_jsonable(env_summary), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Trace Summary")
    lines.append("```json")
    lines.append(json.dumps(to_jsonable(redact_trace_for_report(trace_summary)), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Runtime Contract")
    for item in spec:
        lines.append(f"- {item}")
    lines.append("")
    final_feedback = build_feedback(results, trace_summary, llm=llm, log=log)
    lines.append("## Current Defects And Guidance")
    lines.append("```json")
    lines.append(json.dumps(to_jsonable(final_feedback), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Candidate Iterations")
    for result in results:
        lines.append(f"### Iter {result.iteration}: {result.name}")
        lines.append(f"- Strategy: {result.strategy}")
        lines.append(f"- Expected benefit: {result.expected_benefit}")
        lines.append(f"- Risk: {result.risk}")
        lines.append(f"- Precheck: {result.precheck_ok}")
        lines.append(f"- Official correctness: {result.correctness_ok}")
        lines.append(f"- Custom stress: {result.stress_ok}")
        lines.append(f"- Benchmark: {result.benchmark_ok}")
        lines.append(f"- Score: {result.score:.3f}")
        if result.llm_notes:
            lines.append(f"- LLM/self-check notes: {result.llm_notes}")
        if result.benchmark:
            lines.append("- Benchmark rows:")
            for row in result.benchmark:
                lines.append(
                    "  - "
                    f"{row.get('case_name')}: total={row.get('tokens_per_second', 0):.2f} tok/s, "
                    f"decode={row.get('decode_tokens_per_second', 0):.2f} tok/s, "
                    f"peak={row.get('peak_memory_mb', 0):.2f} MB"
                )
        if result.failure_reason:
            compact = result.failure_reason.replace("\n", " ")[-1200:]
            lines.append(f"- Failure reason: {compact}")
        lines.append("")
    lines.append("## LLM Integration")
    llm_events = [event for event in log.events if event["module"] == "llm"]
    if llm_events:
        for event in llm_events:
            lines.append(f"- {event['time']}: {event['message']}")
            if event.get("data"):
                lines.append("```json")
                lines.append(json.dumps(to_jsonable(event["data"]), indent=2, ensure_ascii=False))
                lines.append("```")
    else:
        lines.append("- No LLM event was recorded.")
    lines.append("")
    lines.append("## Full Agent Event Log")
    for event in log.events:
        lines.append(f"- [{event['time']}] [{event['module']}] {event['message']}")
    lines.append("")
    (WORKSPACE / "output3.md").write_text("\n".join(lines), encoding="utf-8")
    log.log("report", "wrote /workspace/output3.md", {"path": WORKSPACE / "output3.md"})
