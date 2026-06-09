from __future__ import annotations

import json
from typing import Any

from common import AgentLog, CandidateResult, WORKSPACE, to_jsonable
from feedback import build_feedback


BENCHMARK_CASE_CHARACTERISTICS = [
    "Case 1: prefill-focused workload.",
    "Case 2: decode-focused workload with repeated generation.",
    "Case 3: mixed lifecycle workload with varied sequence lengths.",
    "Case 4: mixed lifecycle workload with relatively uniform sequence lengths.",
]


CASE_NAME_REPLACEMENTS = {
    "case1_sessions_1_4_prefill_4x128": "Case 1",
    "case2_sessions_5_8_decode_8x128x16": "Case 2",
    "case3_sessions_9_12_mixed_64_128_32": "Case 3",
    "case4_sessions_13_16_mixed_128_all": "Case 4",
}


def sanitize_report_value(value: Any) -> Any:
    if isinstance(value, str):
        if "Observed prompt lengths include" in value:
            return "Observed workloads contain varied sequence lengths; preserve safe grouped prefill behavior."
        for private_name, public_name in CASE_NAME_REPLACEMENTS.items():
            value = value.replace(private_name, public_name)
        return value
    if isinstance(value, list):
        return [sanitize_report_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_report_value(item) for key, item in value.items()}
    return value


def feedback_for_report(feedback: dict[str, Any]) -> dict[str, Any]:
    public_fields = ("source", "summary", "priority", "defects", "guidance", "risk_notes")
    public = {key: feedback[key] for key in public_fields if feedback.get(key)}
    return sanitize_report_value(public)


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
    lines.append("- Completed.")
    lines.append("")
    lines.append("## Trace Summary")
    lines.append("- Completed.")
    lines.append("")
    lines.append("## Benchmark Workloads")
    for characteristic in BENCHMARK_CASE_CHARACTERISTICS:
        lines.append(f"- {characteristic}")
    lines.append("")
    lines.append("## Runtime Contract")
    for item in spec:
        lines.append(f"- {item}")
    lines.append("")
    final_feedback = build_feedback(results, trace_summary, llm=llm, log=log)
    lines.append("## Current Defects And Guidance")
    lines.append("```json")
    lines.append(json.dumps(to_jsonable(feedback_for_report(final_feedback)), indent=2, ensure_ascii=False))
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
        if result.failure_reason:
            if result.failure_reason.lower().startswith("benchmark"):
                lines.append("- Failure reason: Benchmark failed; detailed evaluator output omitted.")
            else:
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
