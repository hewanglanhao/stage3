from __future__ import annotations

import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any

from common import AgentLog, CandidateResult, to_jsonable


class LLMClient:
    def __init__(self, log: AgentLog) -> None:
        self.log = log
        self.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.model = os.getenv("BASE_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("MODEL")
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
        self.feedback_timeout = float(os.getenv("LLM_FEEDBACK_TIMEOUT_SECONDS", str(self.timeout)))
        self.candidate_timeout = float(os.getenv("LLM_CANDIDATE_TIMEOUT_SECONDS", str(max(self.timeout, 240.0))))
        self.retries = max(0, int(os.getenv("LLM_RETRIES", "1")))
        self.enabled = bool(self.api_key and self.model and os.getenv("AGENT_USE_LLM", "1") != "0")

    def ask_for_candidate(self, prompt: str) -> dict[str, str] | None:
        content = self._chat_completion(
            system_prompt="You are an expert LLM inference runtime engineer. Return only the requested fixed-format answer.",
            user_prompt=prompt,
            max_tokens=6000,
            temperature=0.2,
            purpose="engine candidate",
            timeout=self.candidate_timeout,
        )
        if content is None:
            return None
        parsed = parse_llm_response(content)
        self.log.log("llm", "received LLM optimization proposal", {
            "strategy": parsed.get("strategy", ""),
            "expected_benefit": parsed.get("expected_benefit", ""),
            "risk": parsed.get("risk", ""),
            "self_check_notes": parsed.get("self_check_notes", ""),
        })
        return parsed

    def ask_for_feedback(self, prompt: str) -> dict[str, Any] | None:
        content = self._chat_completion(
            system_prompt=(
                "You are a strict local-judge analyst for LLM inference runtimes. "
                "Summarize defects and produce actionable optimization guidance. "
                "Return strict JSON only."
            ),
            user_prompt=prompt,
            max_tokens=1800,
            temperature=0.1,
            purpose="feedback summary",
            timeout=self.feedback_timeout,
        )
        if content is None:
            return None
        parsed = extract_json_object(content)
        if parsed is None:
            self.log.log("llm", "LLM feedback response was not valid JSON", {"response_excerpt": content[:1200]})
            return None
        return parsed

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        purpose: str,
        timeout: float | None = None,
    ) -> str | None:
        if not self.enabled:
            self.log.log("llm", f"LLM client disabled or missing API_KEY/BASE_MODEL for {purpose}")
            return None
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.log.log("llm", f"sending LLM request for {purpose}", {
            "model": self.model,
            "base_url": self.base_url,
            "timeout_s": float(timeout if timeout is not None else self.timeout),
            "max_tokens": max_tokens,
            "prompt_chars": len(system_prompt) + len(user_prompt),
            "retries": self.retries,
        })
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": os.getenv(
                    "LLM_USER_AGENT",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
                ),
            },
            method="POST",
        )
        request_timeout = float(timeout if timeout is not None else self.timeout)
        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
                return obj["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                if retryable and attempt < attempts:
                    self.log.log("llm", f"LLM request attempt {attempt} failed for {purpose}; retrying", {
                        "status": exc.code,
                        "reason": exc.reason,
                        "timeout_s": request_timeout,
                    })
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                self.log.log("llm", f"LLM request failed for {purpose}; continuing with fallback", {
                    "status": exc.code,
                    "reason": exc.reason,
                    "timeout_s": request_timeout,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "body_excerpt": redact_secret_text(body[:1200]),
                })
                return None
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < attempts:
                    self.log.log("llm", f"LLM request attempt {attempt} timed out/failed for {purpose}; retrying", {
                        "error": redact_secret_text(repr(exc)),
                        "timeout_s": request_timeout,
                    })
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                self.log.log("llm", f"LLM request failed for {purpose}; continuing with fallback", {
                    "error": redact_secret_text(repr(exc)),
                    "timeout_s": request_timeout,
                    "attempts": attempts,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                })
                return None
            except (KeyError, json.JSONDecodeError) as exc:
                self.log.log("llm", f"LLM response parse failed for {purpose}; continuing with fallback", {
                    "error": redact_secret_text(repr(exc)),
                    "timeout_s": request_timeout,
                })
                return None
        return None


def parse_llm_response(text: str) -> dict[str, str]:
    fields = ["strategy", "expected_benefit", "risk", "patch_or_full_engine", "self_check_notes"]
    result = {field: "" for field in fields}
    pattern = re.compile(r"^(strategy|expected_benefit|risk|patch_or_full_engine|self_check_notes):\s*$", re.M)
    matches = list(pattern.finditer(text))
    if not matches:
        result["patch_or_full_engine"] = text
        return result
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1)] = text[start:end].strip()
    return result


def extract_json_object(text: str) -> dict[str, Any] | None:
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    candidates = fences + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _end = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def extract_python_code(text: str) -> str | None:
    if not text:
        return None
    fences = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S)
    candidates = fences or [text]
    for candidate in candidates:
        if "def create_engine" in candidate and "class Engine" in candidate:
            return candidate.strip() + "\n"
    return None


AGGRESSIVE_RUNTIME_STRATEGY = textwrap.dedent("""
Preferred aggressive optimization path once correctness is already passing:
- Fuse q/k/v projection weights during Engine initialization, run one F.linear per layer for QKV, then split into q/k/v views. Remove or avoid duplicate GPU copies of the original q/k/v weights after the fused weight is built.
- Fuse gate/up projection weights during Engine initialization, run one F.linear for the MLP input projection, then split gate/up before SiLU multiplication and down projection.
- Keep the existing correct KV-cache implementation as a fallback, but prefer a shared packed KV block for requests that came from the same batched prefill/decode group. RequestState may track length, a shared cache block, and row index.
- In decode, if active requests share the same cache block, gather rows with index_select or use the whole block directly instead of concatenating many per-request slices. Fall back to the generic slice/concat path for heterogeneous cache blocks.
- After batched decode, repack selected requests into one updated shared cache block and update each RequestState row, while preserving request_ids output order.
- Preserve prefill replacement, remove semantics, non-contiguous/unsorted request ids, mixed prompt lengths, RoPE positions, dtype behavior, and dynamic config-derived dimensions.
- Implement fused projections first, then shared KV packing. If a risky optimization complicates correctness, keep it behind the proven fallback path.
""").strip()


def build_llm_prompt(
    env_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    spec: list[str],
    results: list[CandidateResult],
    best_code_excerpt: str,
    feedback: dict[str, Any],
    llm_round: int,
    branch_mode: str = "general",
) -> str:
    candidate_summary = [
        {
            "iteration": r.iteration,
            "name": r.name,
            "correctness_ok": r.correctness_ok,
            "stress_ok": r.stress_ok,
            "benchmark_ok": r.benchmark_ok,
            "score": r.score,
            "failure_reason": r.failure_reason[-800:],
            "benchmark": r.benchmark,
        }
        for r in results
    ]
    branch_instructions = build_branch_instructions(branch_mode, trace_summary)
    prompt_trace_summary = trace_summary if branch_mode == "token_aware" else strip_token_profile(trace_summary)
    return textwrap.dedent(f"""
    LLM optimization round: {llm_round}
    Branch mode: {branch_mode}

    Branch-specific instructions:
    {branch_instructions}

    Preferred aggressive runtime strategy:
    {AGGRESSIVE_RUNTIME_STRATEGY}

    Current model/environment summary:
    {json.dumps(to_jsonable(env_summary), indent=2, ensure_ascii=False)[:5000]}

    Observed trace/benchmark summary:
    {json.dumps(to_jsonable(prompt_trace_summary), indent=2, ensure_ascii=False)[:4000]}

    Runtime hard constraints:
    {json.dumps(spec, indent=2, ensure_ascii=False)}

    Candidate results so far:
    {json.dumps(to_jsonable(candidate_summary), indent=2, ensure_ascii=False)[:6000]}

    LLM-summarized defects and next-step guidance from feedback.py:
    {json.dumps(to_jsonable(feedback), indent=2, ensure_ascii=False)[:6000]}

    Current best engine excerpt:
    ```python
    {best_code_excerpt[:5000]}
    ```

    Goal: use the defects/guidance above to produce the next full engine.py. First preserve correctness, then aggressively improve decode/mixed throughput using the fused-projection and shared-KV strategy when applicable. Keep the public interface unchanged, read config dynamically, preserve request state semantics, and avoid hard-coded model dimensions. The existing optimized_torch candidate remains in the agent selection pool as a fallback, so an experimental candidate must still pass local correctness before it can be selected.

    Return exactly this format. Provide a full engine.py implementation inside patch_or_full_engine, not a diff:
    strategy:
    expected_benefit:
    risk:
    patch_or_full_engine:
    self_check_notes:
    """).strip()


def build_branch_instructions(branch_mode: str, trace_summary: dict[str, Any]) -> str:
    if branch_mode != "token_aware":
        return textwrap.dedent("""
        General optimization branch. Improve the current best engine using benchmark feedback,
        without relying on token content beyond normal runtime input_ids handling.

        For a performance-passing baseline, prefer structural runtime changes: fused QKV
        projection, fused gate/up projection, and shared packed KV cache blocks for batched
        prefill/decode groups. Keep a generic fallback path for mixed cache ownership.
        """).strip()

    token_profile = trace_summary.get("token_content_profile", {})
    return textwrap.dedent(f"""
    Token-aware experimental branch. The user has explicitly permitted sending this redacted
    token-content profile to the external LLM. Raw token ids are intentionally not included.

    Redacted token-content profile:
    {json.dumps(to_jsonable(token_profile), indent=2, ensure_ascii=False)[:4500]}

    Allowed token-aware ideas:
    - Combine the aggressive fused-projection/shared-KV strategy with token-profile observations; token content should guide grouping/cache sizing, not replace the model computation.
    - Add generic exact-input content-hash caches for prefill states/logits when the same full prompt appears again; always fall back to the normal KV-cache path on cache miss.
    - Use token-profile observations to tune preallocation/cache sizing and grouping decisions, while deriving all model dimensions from config.
    - Branch on runtime-observed input_ids shapes and non-reversible content hashes only when the branch computes exactly the same logits and preserves request order.
    - Keep the optimized grouped KV-cache behavior as the default path for ordinary prompts and as the fallback if shared KV ownership is heterogeneous.

    Forbidden shortcuts:
    - Do not hard-code raw token ids, logits, model outputs, hidden dimensions, or request ids.
    - Do not print or log raw token ids.
    - Do not return stale logits for a request unless the full prompt/token history matches exactly.
    """).strip()


def strip_token_profile(trace_summary: dict[str, Any]) -> dict[str, Any]:
    if "token_content_profile" not in trace_summary:
        return trace_summary
    stripped = dict(trace_summary)
    profile = trace_summary.get("token_content_profile") or {}
    stripped["token_content_profile"] = {
        "available": bool(profile.get("available")) if isinstance(profile, dict) else False,
        "raw_token_values_redacted": True,
        "omitted_from_general_branch_prompt": True,
        "observations": profile.get("observations", []) if isinstance(profile, dict) else [],
    }
    return stripped


def redact_secret_text(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer REDACTED", text)
    return text
