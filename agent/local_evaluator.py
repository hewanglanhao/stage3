from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

from common import AgentLog, CandidateResult, EVALUATOR_DIR, resolve_device, run_subprocess


def precheck_candidate(candidate: CandidateResult, log: AgentLog) -> None:
    code, output = run_subprocess([sys.executable, "-m", "py_compile", str(candidate.path)], timeout=30)
    if code != 0:
        candidate.failure_reason = "py_compile failed:\n" + output[-4000:]
        log.log("precheck", f"candidate {candidate.name} failed syntax check", {"output": output[-2000:]})
        return
    import_code = textwrap.dedent(
        f"""
        import importlib.util
        path = {str(candidate.path)!r}
        spec = importlib.util.spec_from_file_location('candidate_engine', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, 'create_engine'), 'missing create_engine'
        print('precheck_import_ok')
        """
    )
    code, output = run_subprocess([sys.executable, "-c", import_code], timeout=30)
    if code != 0:
        candidate.failure_reason = "import/create_engine precheck failed:\n" + output[-4000:]
        log.log("precheck", f"candidate {candidate.name} failed import check", {"output": output[-2000:]})
        return
    candidate.precheck_ok = True
    log.log("precheck", f"candidate {candidate.name} passed precheck", {"path": candidate.path})


def run_official_correctness(candidate: CandidateResult, model_config_path: Path, weight_dir: Path, device: str, log: AgentLog) -> None:
    if not candidate.precheck_ok:
        return
    cmd = [
        sys.executable,
        str(EVALUATOR_DIR / "test_correctness.py"),
        "--engine",
        str(candidate.path),
        "--model-config",
        str(model_config_path),
        "--weight-dir",
        str(weight_dir),
        "--device",
        device,
    ]
    code, output = run_subprocess(cmd, timeout=int(os.getenv("AGENT_CORRECTNESS_TIMEOUT", "180")))
    candidate.correctness_output = output
    if code == 0:
        candidate.correctness_ok = True
        log.log("correctness", f"candidate {candidate.name} passed official correctness", {"output": output[-1500:]})
    else:
        candidate.failure_reason = "official correctness failed:\n" + output[-5000:]
        log.log("correctness", f"candidate {candidate.name} failed official correctness", {"output": output[-2500:]})


def run_custom_stress(candidate: CandidateResult, model_config: dict[str, Any], weight_dir: Path, device: str, log: AgentLog) -> None:
    if not candidate.correctness_ok:
        return
    try:
        import torch

        if str(EVALUATOR_DIR) not in sys.path:
            sys.path.insert(0, str(EVALUATOR_DIR))
        from reference_model import ReferenceModel  # type: ignore

        actual_device = resolve_device(device)
        torch.manual_seed(101)
        vocab_size = int(model_config["vocab_size"])
        spec = importlib.util.spec_from_file_location(f"candidate_engine_{candidate.iteration}", candidate.path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        engine = mod.create_engine(model_config, str(weight_dir), actual_device)
        ref_model = ReferenceModel(model_config, str(weight_dir), actual_device)
        request_tokens: dict[int, Any] = {}

        def ref_last(ids: Any) -> Any:
            return ref_model.forward(ids.unsqueeze(0))[0, -1, :]

        def assert_close(case: str, student: Any, expected: Any) -> None:
            student_cpu = student.detach().float().cpu()
            expected_cpu = expected.detach().float().cpu()
            if student_cpu.shape != expected_cpu.shape or not torch.allclose(student_cpu, expected_cpu, atol=1e-2, rtol=1e-2):
                diff = (student_cpu - expected_cpu).abs()
                raise AssertionError(
                    f"{case}: shape={tuple(student_cpu.shape)} expected={tuple(expected_cpu.shape)} max_abs={float(diff.max()):.6g}"
                )

        def prefill(case: str, request_ids: list[int], lengths: list[int]) -> None:
            input_ids = [torch.randint(0, vocab_size, (length,), device=actual_device, dtype=torch.long) for length in lengths]
            student = engine.prefill(request_ids, input_ids)
            expected = []
            for rid, ids in zip(request_ids, input_ids):
                request_tokens[int(rid)] = ids.clone()
                expected.append(ref_last(ids))
            assert_close(case, student, torch.stack(expected, dim=0))

        def decode(case: str, request_ids: list[int]) -> None:
            token_ids = torch.randint(0, vocab_size, (len(request_ids),), device=actual_device, dtype=torch.long)
            student = engine.decode(request_ids, token_ids)
            expected = []
            for rid, tok in zip(request_ids, token_ids):
                rid = int(rid)
                request_tokens[rid] = torch.cat([request_tokens[rid], tok.reshape(1)])
                expected.append(ref_last(request_tokens[rid]))
            assert_close(case, student, torch.stack(expected, dim=0))

        def remove(request_ids: list[int]) -> None:
            engine.remove(request_ids)
            for rid in request_ids:
                request_tokens.pop(int(rid), None)

        with torch.inference_mode():
            prefill("stress_variable_prefill", [101, 7], [3, 9])
            decode("stress_decode_reordered", [7, 101])
            prefill("stress_replace_prefill", [101], [5])
            decode("stress_decode_after_replace", [101, 7])
            prefill("stress_noncontiguous_insert", [42, 5, 77], [1, 4, 6])
            remove([7, 42])
            decode("stress_decode_after_remove", [77, 5, 101])
            remove([5, 77, 101])

        candidate.stress_ok = True
        candidate.stress_output = "custom stress passed"
        log.log("stress", f"candidate {candidate.name} passed custom stress cases")
        del engine
        del ref_model
        if actual_device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as exc:
        candidate.stress_output = traceback.format_exc()
        candidate.failure_reason = "custom stress failed:\n" + candidate.stress_output[-5000:]
        log.log("stress", f"candidate {candidate.name} failed custom stress", {"error": repr(exc), "traceback": candidate.stress_output[-2500:]})


def run_benchmark(candidate: CandidateResult, model_config_path: Path, weight_dir: Path, device: str, log: AgentLog) -> None:
    if not (candidate.correctness_ok and candidate.stress_ok):
        return
    repeat = os.getenv("AGENT_BENCH_REPEAT", "2")
    warmup = os.getenv("AGENT_BENCH_WARMUP", "0")
    cmd = [
        sys.executable,
        str(EVALUATOR_DIR / "benchmark_throughput.py"),
        "--engine",
        str(candidate.path),
        "--model-config",
        str(model_config_path),
        "--weight-dir",
        str(weight_dir),
        "--device",
        device,
        "--warmup",
        warmup,
        "--repeat",
        repeat,
    ]
    code, output = run_subprocess(cmd, timeout=int(os.getenv("AGENT_BENCH_TIMEOUT", "240")))
    if code != 0:
        candidate.failure_reason = "benchmark failed:\n" + output[-5000:]
        log.log("benchmark", f"candidate {candidate.name} failed benchmark", {"output": output[-2500:]})
        return
    try:
        candidate.benchmark = parse_json_array_from_output(output)
        candidate.benchmark_ok = True
        candidate.score = score_benchmark(candidate.benchmark)
        log.log("benchmark", f"candidate {candidate.name} benchmark complete", {
            "score": candidate.score,
            "benchmark": candidate.benchmark,
        })
    except Exception as exc:
        candidate.failure_reason = "benchmark JSON parse failed:\n" + output[-5000:]
        log.log("benchmark", f"candidate {candidate.name} benchmark parse failed", {"error": repr(exc), "output": output[-2500:]})


def parse_json_array_from_output(output: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", output):
        try:
            obj, _end = decoder.raw_decode(output[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
    raise json.JSONDecodeError("no JSON array found", output, 0)


def score_benchmark(rows: list[dict[str, Any]]) -> float:
    by_case = {str(row.get("case_name")): row for row in rows}

    def total_tps(case_name: str) -> float:
        return float(by_case.get(case_name, {}).get("tokens_per_second", 0.0) or 0.0)

    def decode_tps(case_name: str) -> float:
        return float(by_case.get(case_name, {}).get("decode_tokens_per_second", 0.0) or 0.0)

    case1_prefill = total_tps("case1_sessions_1_4_prefill_4x128")
    case2_total = total_tps("case2_sessions_5_8_decode_8x128x16")
    case2_decode = decode_tps("case2_sessions_5_8_decode_8x128x16")
    case3_mixed = total_tps("case3_sessions_9_12_mixed_64_128_32")
    case4_mixed = total_tps("case4_sessions_13_16_mixed_128_all")

    if not any((case1_prefill, case2_total, case3_mixed, case4_mixed)):
        # Backward-compatible fallback for older benchmark_throughput.py output.
        case1_prefill = total_tps("prefill")
        case2_decode = decode_tps("decode")
        case3_mixed = total_tps("mixed")
        case4_mixed = case3_mixed

    peak = max((float(row.get("peak_memory_mb", 0.0) or 0.0) for row in rows), default=0.0)
    memory_penalty = peak * 0.0001
    return (
        case1_prefill * 0.20
        + case2_total * 0.60
        + case2_decode * 1.00
        + case3_mixed * 1.50
        + case4_mixed * 1.50
        - memory_penalty
    )


def evaluate_candidate(candidate: CandidateResult, model_config_path: Path, weight_dir: Path, model_config: dict[str, Any], device: str, log: AgentLog) -> None:
    log.log("candidate", f"evaluating iter {candidate.iteration}: {candidate.name}", {
        "strategy": candidate.strategy,
        "expected_benefit": candidate.expected_benefit,
        "risk": candidate.risk,
    })
    precheck_candidate(candidate, log)
    run_official_correctness(candidate, model_config_path, weight_dir, device, log)
    run_custom_stress(candidate, model_config, weight_dir, device, log)
    run_benchmark(candidate, model_config_path, weight_dir, device, log)
