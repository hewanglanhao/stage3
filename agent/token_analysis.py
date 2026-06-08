from __future__ import annotations

import math
import sys
from collections import Counter
from typing import Any

from common import AgentLog, SOURCE_ROOT


def summarize_token_content(model_config: dict[str, Any], log: AgentLog) -> dict[str, Any]:
    """Build a redacted token-content profile for token-aware optimization prompts.

    The profile intentionally contains only aggregate counts and ratios. It never
    returns token ids, full prompts, decode-token sequences, or source paths.
    """
    if not token_profile_enabled():
        summary = {
            "available": False,
            "reason": "disabled_by_AGENT_TOKEN_PROFILE",
            "raw_token_values_redacted": True,
        }
        log.log("token_analysis", "token-content profile disabled", summary)
        return summary

    fixture = load_fixture()
    if not fixture:
        summary = {
            "available": False,
            "reason": "no built-in benchmark token profile available",
            "raw_token_values_redacted": True,
        }
        log.log("token_analysis", "no token-content profile available", summary)
        return summary

    vocab_size = int(model_config.get("vocab_size", 0) or 0)
    case_profiles = []
    prompt_signatures = []
    decode_signatures = []

    for case_name, events in fixture.items():
        prefill_tokens: list[int] = []
        decode_tokens: list[int] = []
        prompt_lengths: list[int] = []
        ops = Counter()
        local_prompt_signatures = []
        local_decode_signatures = []

        for event in events:
            op = str(event.get("op"))
            ops[op] += 1
            if op == "prefill":
                for row in event.get("input_ids", []):
                    row_tokens = [int(x) for x in row]
                    prefill_tokens.extend(row_tokens)
                    prompt_lengths.append(len(row_tokens))
                    sig = stable_shape_signature(row_tokens)
                    local_prompt_signatures.append(sig)
                    prompt_signatures.append(sig)
            elif op == "decode":
                row_tokens = [int(x) for x in event.get("token_ids", [])]
                decode_tokens.extend(row_tokens)
                sig = stable_shape_signature(row_tokens)
                local_decode_signatures.append(sig)
                decode_signatures.append(sig)

        case_profiles.append({
            "case_name": case_name,
            "ops": dict(ops),
            "prompt_lengths": sorted(set(prompt_lengths)),
            "prefill": token_stats(prefill_tokens, vocab_size),
            "decode": token_stats(decode_tokens, vocab_size),
            "combined": token_stats(prefill_tokens + decode_tokens, vocab_size),
            "duplicate_prompt_ratio_within_case": duplicate_ratio(local_prompt_signatures),
            "duplicate_decode_step_ratio_within_case": duplicate_ratio(local_decode_signatures),
        })

    summary = {
        "available": True,
        "raw_token_values_redacted": True,
        "case_profiles": case_profiles,
        "cross_case_duplicate_prompt_ratio": duplicate_ratio(prompt_signatures),
        "cross_case_duplicate_decode_step_ratio": duplicate_ratio(decode_signatures),
        "observations": infer_observations(case_profiles),
        "safe_token_aware_branches": [
            "Use runtime input_ids content hashes to cache/reuse exact prompt states only when the full token sequence matches, with a generic fallback for all other prompts.",
            "Use token-profile information to choose conservative cache sizes and preallocation paths without hard-coding model dimensions or request ids.",
            "Keep the existing grouped KV-cache engine as the fallback path if a content-aware fast path misses or becomes unsafe.",
        ],
        "forbidden_shortcuts": [
            "Do not hard-code logits or model outputs.",
            "Do not rely on request id values for correctness.",
            "Do not print, log, or expose raw token ids.",
        ],
    }
    log.log("token_analysis", "built redacted token-content profile for optimization", {
        "available": True,
        "case_count": len(case_profiles),
        "raw_token_values_redacted": True,
        "observations": summary["observations"],
    })
    return summary


def token_profile_enabled() -> bool:
    import os

    return os.getenv("AGENT_TOKEN_PROFILE", "1") != "0"


def load_fixture() -> dict[str, Any]:
    evaluator_dir = SOURCE_ROOT / "evaluator"
    if str(evaluator_dir) not in sys.path:
        sys.path.insert(0, str(evaluator_dir))
    try:
        from benchmark_token_fixture import BENCHMARK_TOKEN_FIXTURE  # type: ignore
    except Exception:
        return {}
    return BENCHMARK_TOKEN_FIXTURE if isinstance(BENCHMARK_TOKEN_FIXTURE, dict) else {}


def stable_shape_signature(tokens: list[int]) -> tuple[int, int, int, int]:
    # A compact non-reversible signature for duplicate detection only.
    h1 = 1469598103934665603
    h2 = 1099511628211
    for token in tokens:
        value = int(token) & 0xFFFFFFFF
        h1 ^= value
        h1 = (h1 * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        h2 = (h2 + value * 16777619) & 0xFFFFFFFFFFFFFFFF
    return (len(tokens), h1, h2, sum(tokens) & 0xFFFFFFFF)


def duplicate_ratio(signatures: list[tuple[int, int, int, int]]) -> float:
    if not signatures:
        return 0.0
    return round((len(signatures) - len(set(signatures))) / len(signatures), 4)


def token_stats(tokens: list[int], vocab_size: int) -> dict[str, Any]:
    total = len(tokens)
    if total == 0:
        return {"count": 0}
    counts = Counter(tokens)
    expected_unique = None
    unique_vs_uniform = None
    if vocab_size > 0:
        expected_unique = vocab_size * (1.0 - (1.0 - 1.0 / vocab_size) ** total)
        unique_vs_uniform = len(counts) / expected_unique if expected_unique else None
    entropy = -sum((freq / total) * math.log2(freq / total) for freq in counts.values())
    return {
        "count": total,
        "unique": len(counts),
        "unique_ratio": round(len(counts) / total, 4),
        "expected_unique_uniform": round(expected_unique, 1) if expected_unique is not None else None,
        "unique_vs_uniform_ratio": round(unique_vs_uniform, 4) if unique_vs_uniform is not None else None,
        "top1_freq_ratio": round(counts.most_common(1)[0][1] / total, 4),
        "top10_freq_ratio": round(sum(freq for _token, freq in counts.most_common(10)) / total, 4),
        "adjacent_repeat_ratio": round(sum(1 for a, b in zip(tokens, tokens[1:]) if a == b) / max(1, total - 1), 4),
        "low_100_ratio": round(sum(1 for token in tokens if token < 100) / total, 4),
        "low_1000_ratio": round(sum(1 for token in tokens if token < 1000) / total, 4),
        "upper_half_ratio": round(sum(1 for token in tokens if vocab_size and token >= vocab_size // 2) / total, 4) if vocab_size else None,
        "entropy_bits": round(entropy, 3),
    }


def infer_observations(case_profiles: list[dict[str, Any]]) -> list[str]:
    observations = []
    near_uniform = True
    no_adjacent_repeats = True
    for profile in case_profiles:
        combined = profile.get("combined", {})
        ratio = combined.get("unique_vs_uniform_ratio")
        if ratio is not None and not (0.95 <= float(ratio) <= 1.05):
            near_uniform = False
        if float(combined.get("adjacent_repeat_ratio", 0.0) or 0.0) > 0.01:
            no_adjacent_repeats = False
    if near_uniform:
        observations.append("Aggregate token-id frequencies are close to uniform for the measured fixture; token values alone are unlikely to change matmul cost.")
    if no_adjacent_repeats:
        observations.append("Adjacent repeated token ids are effectively absent; repeat-token shortcuts are unlikely to help.")
    observations.append("Content-aware candidates should focus on exact prompt/decode-sequence caches with safe fallback, while preserving the optimized KV-cache branch.")
    return observations
