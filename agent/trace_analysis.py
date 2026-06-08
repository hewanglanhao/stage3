from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from common import AgentLog, ROOT, SOURCE_ROOT


def analyze_trace(log: AgentLog) -> dict[str, Any]:
    trace_paths = [
        ROOT / "engine_trace.md",
        ROOT / "workspace" / "engine_trace.md",
        SOURCE_ROOT / "doc" / "test_real.md",
        Path("/workspace/real_test/test.md"),
    ]
    events = []
    used_paths = []
    for path in trace_paths:
        if not path.exists():
            continue
        used_paths.append(str(path))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 7 or cells[0] in ("#", "---:"):
                continue
            op = cells[1].strip("`")
            if op not in {"prefill", "decode", "remove"}:
                continue
            try:
                batch_size = int(cells[2])
            except ValueError:
                continue
            events.append({
                "op": op,
                "batch_size": batch_size,
                "request_order": literal_or_text(cells[3]),
                "prompt_lengths": literal_or_text(cells[4]),
                "decode_steps_after_call": literal_or_text(cells[5]),
                "active_lengths_after_call": literal_or_text(cells[6]),
            })

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
        "sample_events": events[:12],
    }
    log.log("trace", "trace analysis summary", summary)
    return summary


def literal_or_text(text: str) -> Any:
    if text in {"None", "`None`", ""}:
        return None
    clean = text.strip("`")
    try:
        return ast.literal_eval(clean)
    except Exception:
        return clean
