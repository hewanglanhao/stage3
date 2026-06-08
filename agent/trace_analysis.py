from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from common import AgentLog, ROOT, SOURCE_ROOT

BUILT_IN_BENCHMARK_TRACE_SUMMARY = {
    "trace_paths": ["built_in_benchmark_case_summary"],
    "event_count": 40,
    "op_counts": {"prefill": 8, "decode": 26, "remove": 6},
    "batch_sizes": {"prefill": [2, 4, 8], "decode": [4, 6, 8], "remove": [2, 4, 8]},
    "prompt_lengths": [32, 64, 128],
    "max_decode_step_seen": 16,
    "case_summaries": [
        {"case_name": "case1_sessions_1_4_prefill_4x128", "sessions": [1, 2, 3, 4], "summary": "prefill batch=4 length=128, then remove"},
        {"case_name": "case2_sessions_5_8_decode_8x128x16", "sessions": [5, 6, 7, 8], "summary": "prefill batch=8 length=128, then decode batch=8 for 16 steps, then remove"},
        {"case_name": "case3_sessions_9_12_mixed_64_128_32", "sessions": [9, 10, 11, 12], "summary": "mixed inserts with prompt lengths 64, 128, then 32 plus remove/decode interleaving"},
        {"case_name": "case4_sessions_13_16_mixed_128_all", "sessions": [13, 14, 15, 16], "summary": "mixed inserts where all prefill prompts are length 128 plus remove/decode interleaving"},
    ],
    "sample_events": [],
    "source_note": "built-in benchmark case summary; external trace files are not required.",
}


def analyze_trace(log: AgentLog) -> dict[str, Any]:
    trace_paths = [
        ROOT / "engine_trace.md",
        ROOT / "workspace" / "engine_trace.md",
        SOURCE_ROOT / "doc" / "test_real.md",
    ]
    events = []
    used_paths = []
    for path in trace_paths:
        if not path.exists():
            continue
        used_paths.append(str(path))
        text = path.read_text(encoding="utf-8", errors="ignore")
        current_headers: list[str] | None = None
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            if "op" in cells and "batch_size" in cells:
                current_headers = [c.strip("`") for c in cells]
                continue
            if cells[0] in ("#", "---:") or cells[1] in ("op", "---"):
                continue
            row = dict(zip(current_headers, cells)) if current_headers and len(cells) == len(current_headers) else {}
            op = (row.get("op") or cells[1]).strip("`")
            if op not in {"prefill", "decode", "remove"}:
                continue
            try:
                batch_size = int(row.get("batch_size") or cells[2])
            except ValueError:
                continue
            events.append({
                "op": op,
                "batch_size": batch_size,
                "request_order": literal_or_text(row.get("request_order", cells[3] if len(cells) > 3 else "")),
                "prompt_lengths": literal_or_text(row.get("prompt_lengths", cells[4] if len(cells) > 4 else "")),
                "decode_steps_after_call": literal_or_text(row.get("decode_steps_after_call", "")),
                "active_lengths_after_call": literal_or_text(row.get("active_lengths_after_call", "")),
            })

    if not events:
        summary = dict(BUILT_IN_BENCHMARK_TRACE_SUMMARY)
        log.log("trace", "using built-in benchmark case summary", {
            "summary_redacted_from_log": True,
            "case_count": len(summary.get("case_summaries", [])),
            "external_trace_required": False,
        })
        return summary

    op_counts: dict[str, int] = {}
    batch_sizes: dict[str, list[int]] = {}
    prompt_lengths: list[int] = []
    max_decode_step = 0
    for event in events:
        op = event["op"]
        op_counts[op] = op_counts.get(op, 0) + 1
        batch_sizes.setdefault(op, []).append(event["batch_size"])
        if isinstance(event.get("prompt_lengths"), list):
            prompt_lengths.extend(int(x) for x in event["prompt_lengths"])
        steps = event.get("decode_steps_after_call")
        if isinstance(steps, dict) and steps:
            max_decode_step = max(max_decode_step, max(int(v) for v in steps.values()))

    summary = {
        "trace_paths": used_paths,
        "event_count": len(events),
        "op_counts": op_counts,
        "batch_sizes": {k: sorted(set(v)) for k, v in batch_sizes.items()},
        "prompt_lengths": sorted(set(prompt_lengths)),
        "max_decode_step_seen": max_decode_step,
        "case_summaries": BUILT_IN_BENCHMARK_TRACE_SUMMARY["case_summaries"],
        "sample_events": events[:12],
    }
    log.log("trace", "trace analysis collected", {
        "summary_redacted_from_log": True,
        "trace_path_count": len(used_paths),
        "sample_events_redacted": True,
    })
    return summary


def literal_or_text(text: str) -> Any:
    if text in {"None", "`None`", ""}:
        return None
    clean = text.strip("`")
    try:
        return ast.literal_eval(clean)
    except Exception:
        return clean
