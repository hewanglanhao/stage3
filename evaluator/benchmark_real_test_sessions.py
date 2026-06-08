from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


CASE_GROUPS = {
    "case1_sessions_1_4_prefill_4x128": [1, 2, 3, 4],
    "case2_sessions_5_8_decode_8x128x16": [5, 6, 7, 8],
    "case3_sessions_9_12_mixed_64_128_32": [9, 10, 11, 12],
    "case4_sessions_13_16_mixed_128_all": [13, 14, 15, 16],
}

TABLE_HEADERS = [
    "#",
    "op",
    "batch_size",
    "request_order",
    "prompt_lengths",
    "prefill_input_ids",
    "decode_token_ids",
    "decode_steps_after_call",
    "active_lengths_after_call",
]


@dataclass
class SessionResult:
    elapsed_ms: float
    prefill_tokens: int
    decode_tokens: int
    peak_memory_mb: float


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_student_engine(engine_path: Path):
    spec = importlib.util.spec_from_file_location("student_engine_real_trace", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load engine module from {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def literal_cell(text: str) -> Any:
    clean = text.strip().strip("`")
    if clean in {"", "None"}:
        return None
    return ast.literal_eval(clean)


def parse_trace_file(path: Path) -> dict[int, list[dict[str, Any]]]:
    sessions: dict[int, list[dict[str, Any]]] = {}
    current_session: int | None = None
    current_headers: list[str] | None = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"##\s+Engine Session\s+(\d+)", line)
        if match:
            current_session = int(match.group(1))
            sessions[current_session] = []
            current_headers = None
            continue
        if current_session is None or not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(TABLE_HEADERS):
            continue
        if "op" in cells and "batch_size" in cells:
            current_headers = [cell.strip("`") for cell in cells]
            continue
        if cells[0] in {"#", "---:"} or cells[1] in {"op", "---"}:
            continue
        headers = current_headers or TABLE_HEADERS
        row = dict(zip(headers, cells))
        op = row["op"].strip("`")
        if op not in {"prefill", "decode", "remove"}:
            continue

        request_ids = [int(x) for x in literal_cell(row["request_order"])]
        event: dict[str, Any] = {"op": op, "request_ids": request_ids}
        if op == "prefill":
            token_map = literal_cell(row["prefill_input_ids"])
            if not isinstance(token_map, dict):
                raise ValueError(f"session {current_session}: prefill_input_ids is not a dict")
            event["input_ids"] = [[int(x) for x in token_map[int(rid)]] for rid in request_ids]
        elif op == "decode":
            token_values = literal_cell(row["decode_token_ids"])
            if isinstance(token_values, dict):
                event["token_ids"] = [int(token_values[int(rid)]) for rid in request_ids]
            elif isinstance(token_values, list):
                event["token_ids"] = [int(x) for x in token_values]
            else:
                raise ValueError(f"session {current_session}: decode_token_ids is not a dict/list")
        sessions[current_session].append(event)

    missing = [sid for group in CASE_GROUPS.values() for sid in group if sid not in sessions]
    if missing:
        raise ValueError(f"trace file is missing sessions: {missing}")
    return sessions


def materialize_events(raw_events: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    events = []
    for event in raw_events:
        op = event["op"]
        request_ids = [int(x) for x in event["request_ids"]]
        if op == "prefill":
            events.append({
                "op": "prefill",
                "request_ids": request_ids,
                "input_ids": [
                    torch.tensor(row, dtype=torch.long, device=device)
                    for row in event["input_ids"]
                ],
            })
        elif op == "decode":
            events.append({
                "op": "decode",
                "request_ids": request_ids,
                "token_ids": torch.tensor(event["token_ids"], dtype=torch.long, device=device),
            })
        elif op == "remove":
            events.append({"op": "remove", "request_ids": request_ids})
        else:
            raise ValueError(f"unknown op: {op}")
    return events


def timed_session(engine, raw_events: list[dict[str, Any]], device: str) -> SessionResult:
    events = materialize_events(raw_events, device)
    prefill_tokens = 0
    decode_tokens = 0

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for event in events:
            op = event["op"]
            if op == "prefill":
                engine.prefill(event["request_ids"], event["input_ids"])
                prefill_tokens += sum(int(ids.numel()) for ids in event["input_ids"])
            elif op == "decode":
                engine.decode(event["request_ids"], event["token_ids"])
                decode_tokens += int(event["token_ids"].numel())
            elif op == "remove":
                engine.remove(event["request_ids"])
            else:
                raise ValueError(f"unknown op: {op}")
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    peak_memory_mb = 0.0
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return SessionResult(
        elapsed_ms=elapsed_ms,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        peak_memory_mb=peak_memory_mb,
    )


def benchmark_case(case_name: str, session_ids: list[int], sessions: dict[int, list[dict[str, Any]]], engine_mod, model_config: dict[str, Any], weight_dir: str, device: str) -> dict[str, float | str]:
    elapsed_ms = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    peak_memory_mb = 0.0

    for session_id in session_ids:
        engine = engine_mod.create_engine(model_config, weight_dir, device)
        result = timed_session(engine, sessions[session_id], device)
        elapsed_ms += result.elapsed_ms
        prefill_tokens += result.prefill_tokens
        decode_tokens += result.decode_tokens
        peak_memory_mb = max(peak_memory_mb, result.peak_memory_mb)
        del engine
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    elapsed_s = elapsed_ms / 1000.0
    total_tokens = prefill_tokens + decode_tokens
    return {
        "case_name": case_name,
        "tokens_per_second": total_tokens / elapsed_s if elapsed_s else 0.0,
        "decode_tokens_per_second": decode_tokens / elapsed_s if decode_tokens and elapsed_s else 0.0,
        "peak_memory_mb": peak_memory_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /workspace/engine.py using the 16 sessions in real_test/test.md.")
    parser.add_argument("--engine", default="/workspace/engine.py")
    parser.add_argument("--model-config", default="/workspace/stage3/target/model_config.json")
    parser.add_argument("--weight-dir", default="/workspace/stage3/target/weights")
    parser.add_argument("--trace", default="/workspace/real_test/test.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="/workspace/stage3/output")
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(0)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    sessions = parse_trace_file(Path(args.trace))
    engine_mod = load_student_engine(Path(args.engine))

    results = [
        benchmark_case(case_name, session_ids, sessions, engine_mod, model_config, args.weight_dir, device)
        for case_name, session_ids in CASE_GROUPS.items()
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_real_test_sessions_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "engine": str(Path(args.engine)),
        "trace": str(Path(args.trace)),
        "model_config": str(Path(args.model_config)),
        "weight_dir": str(Path(args.weight_dir)),
        "device": device,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"saved_to": str(output_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
