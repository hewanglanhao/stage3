from __future__ import annotations

import re
from typing import Any

from common import AgentLog, CandidateResult, WORKSPACE


BENCHMARK_CASE_CHARACTERISTICS = [
    "Case 1: prefill-focused workload.",
    "Case 2: decode-focused workload with repeated generation.",
    "Case 3: mixed lifecycle workload with varied sequence lengths.",
    "Case 4: mixed lifecycle workload with relatively uniform sequence lengths.",
]


AGENT_ARCHITECTURE = [
    "Environment and input probe: resolves model inputs and inspects runtime capabilities.",
    "Trace and token analysis: characterizes the workload for optimization without exposing raw token values.",
    "Runtime specification: converts the required engine API and state semantics into hard constraints.",
    "Candidate generation: provides a grouped KV-cache performance baseline and asks the LLM to generate stronger runtimes.",
    "Feedback loop: summarizes correctness and performance results to guide the next LLM candidate.",
    "Local evaluation: runs import checks, correctness tests, request-state stress tests, and benchmarks.",
    "Selection and fallback: keeps the highest-scoring correct candidate and never replaces it with a failed iteration.",
    "Reporting: records the selected strategy, candidate outcomes, LLM activity, and final engine choice.",
]


def compact_text(value: Any) -> str:
    text = re.sub(r"```.*?```", "", str(value or ""), flags=re.S)
    return " ".join(text.replace("`", "").split())


def selected_strategy_summary(best: CandidateResult, llm: object | None, log: AgentLog) -> dict[str, Any]:
    fallback = {
        "summary": compact_text(best.strategy),
        "techniques": [],
        "performance_rationale": compact_text(best.expected_benefit),
        "tradeoffs": compact_text(best.risk),
        "source": "candidate_metadata",
    }
    if llm is None or not getattr(llm, "enabled", False):
        return fallback

    prompt = (
        "Summarize the optimization strategy of the selected LLM inference runtime.\n"
        "Use plain technical prose only. Do not output source code, pseudocode, code fences, "
        "function definitions, implementation snippets, model dimensions, token values, or trace details.\n"
        "Describe only what the strategy does and why it was selected.\n\n"
        f"Candidate name: {best.name}\n"
        f"Original strategy: {compact_text(best.strategy)}\n"
        f"Expected benefit: {compact_text(best.expected_benefit)}\n"
        f"Known risk: {compact_text(best.risk)}\n"
        f"Correctness passed: {best.correctness_ok}\n"
        f"Stress tests passed: {best.stress_ok}\n"
        f"Benchmark passed: {best.benchmark_ok}\n"
        f"Composite score: {best.score:.3f}\n\n"
        "Return strict JSON with this schema: "
        "{\"summary\": \"short paragraph\", "
        "\"techniques\": [\"technique description\"], "
        "\"performance_rationale\": \"short paragraph\", "
        "\"tradeoffs\": \"short paragraph\"}"
    )
    result = llm.ask_for_strategy_summary(prompt)
    if not result:
        log.log("report", "LLM strategy summary unavailable; using candidate metadata")
        return fallback

    techniques = result.get("techniques", [])
    if not isinstance(techniques, list):
        techniques = []
    return {
        "summary": compact_text(result.get("summary")) or fallback["summary"],
        "techniques": [compact_text(item) for item in techniques if compact_text(item)][:8],
        "performance_rationale": compact_text(result.get("performance_rationale")) or fallback["performance_rationale"],
        "tradeoffs": compact_text(result.get("tradeoffs")) or fallback["tradeoffs"],
        "source": "llm_strategy_summary",
    }


def candidate_status(result: CandidateResult) -> str:
    if not result.precheck_ok:
        return "precheck failed"
    if not result.correctness_ok:
        return "correctness failed"
    if not result.stress_ok:
        return "stress failed"
    if not result.benchmark_ok:
        return "benchmark unavailable"
    return "passed"


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
    lines.append("## Agent Architecture")
    for module_description in AGENT_ARCHITECTURE:
        lines.append(f"- {module_description}")
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
    strategy_summary = selected_strategy_summary(best, llm, log)
    lines.append("## Selected Optimization Strategy")
    lines.append(f"- Summary: {strategy_summary['summary']}")
    if strategy_summary["techniques"]:
        lines.append("- Main techniques:")
        for technique in strategy_summary["techniques"]:
            lines.append(f"  - {technique}")
    lines.append(f"- Why selected: {strategy_summary['performance_rationale']}")
    lines.append(f"- Trade-offs: {strategy_summary['tradeoffs']}")
    lines.append("")
    lines.append("## Candidate Iterations")
    for result in results:
        score = f"{result.score:.3f}" if result.benchmark_ok else "n/a"
        lines.append(
            f"- Iter {result.iteration} `{result.name}`: "
            f"{candidate_status(result)}, score={score}; "
            f"strategy: {compact_text(result.strategy)}"
        )
    lines.append("")
    lines.append("## LLM Integration")
    llm_events = [event for event in log.events if event["module"] == "llm"]
    request_count = sum("sending LLM request" in event["message"] for event in llm_events)
    proposal_count = sum("received LLM optimization proposal" in event["message"] for event in llm_events)
    feedback_request_count = sum("sending LLM request for feedback summary" in event["message"] for event in llm_events)
    strategy_summary_request_count = sum("sending LLM request for selected strategy summary" in event["message"] for event in llm_events)
    failure_count = sum(
        any(marker in event["message"].lower() for marker in ("failed", "parse failed", "did not produce"))
        for event in llm_events
    )
    lines.append(f"- LLM requests: {request_count}")
    lines.append(f"- Candidate proposals received: {proposal_count}")
    lines.append(f"- Feedback requests: {feedback_request_count}")
    lines.append(f"- Strategy summary requests: {strategy_summary_request_count}")
    lines.append(f"- Failed or unusable interactions: {failure_count}")
    lines.append("")
    lines.append("## Full Agent Event Log")
    for event in log.events:
        lines.append(f"- [{event['time']}] [{event['module']}] {event['message']}")
    lines.append("")
    (WORKSPACE / "output3.md").write_text("\n".join(lines), encoding="utf-8")
    log.log("report", "wrote /workspace/output3.md", {"path": WORKSPACE / "output3.md"})
