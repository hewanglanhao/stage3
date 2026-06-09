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
from optimization_guidance import PREFERRED_RUNTIME_STRATEGY


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

    def ask_for_strategy_summary(self, prompt: str) -> dict[str, Any] | None:
        content = self._chat_completion(
            system_prompt=(
                "You summarize the selected LLM inference runtime optimization strategy. "
                "Return strict JSON containing plain technical prose only. "
                "Never include source code, pseudocode, code fences, snippets, token values, or trace details."
            ),
            user_prompt=prompt,
            max_tokens=900,
            temperature=0.1,
            purpose="selected strategy summary",
            timeout=self.feedback_timeout,
        )
        if content is None:
            return None
        parsed = extract_json_object(content)
        if parsed is None:
            self.log.log("llm", "LLM strategy summary response was not valid JSON")
            return None
        self.log.log("llm", "received selected strategy summary", {
            "technique_count": len(parsed.get("techniques", [])) if isinstance(parsed.get("techniques"), list) else 0,
            "code_content_excluded": True,
        })
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
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504, 524}
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


def build_llm_prompt(
    env_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    spec: list[str],
    best_result: CandidateResult | None,
    best_code: str,
    feedback: dict[str, Any],
    llm_round: int,
    branch_mode: str = "general",
) -> str:
    best_candidate_summary = None
    if best_result is not None:
        best_candidate_summary = {
            "iteration": best_result.iteration,
            "name": best_result.name,
            "correctness_ok": best_result.correctness_ok,
            "stress_ok": best_result.stress_ok,
            "benchmark_ok": best_result.benchmark_ok,
            "score": best_result.score,
            "failure_reason": best_result.failure_reason[-800:],
            "benchmark": best_result.benchmark,
        }
    feedback_for_prompt = {
        key: value
        for key, value in feedback.items()
        if key not in {"candidate_table", "current_best", "latest_candidate"}
    }
    branch_instructions = build_branch_instructions(branch_mode, trace_summary)
    prompt_trace_summary = trace_summary if branch_mode == "token_aware" else strip_token_profile(trace_summary)
    return textwrap.dedent(f"""
    LLM optimization round: {llm_round}
    Branch mode: {branch_mode}

    Branch-specific instructions:
    {branch_instructions}

    Preferred runtime optimization priorities:
    {PREFERRED_RUNTIME_STRATEGY}

    Current model/environment summary:
    {json.dumps(to_jsonable(env_summary), indent=2, ensure_ascii=False)[:5000]}

    Observed trace/benchmark summary:
    {json.dumps(to_jsonable(prompt_trace_summary), indent=2, ensure_ascii=False)[:4000]}

    Runtime hard constraints:
    {json.dumps(spec, indent=2, ensure_ascii=False)}

    Current best candidate result:
    {json.dumps(to_jsonable(best_candidate_summary), indent=2, ensure_ascii=False)}

    LLM-summarized defects and next-step guidance from feedback.py:
    {json.dumps(to_jsonable(feedback_for_prompt), indent=2, ensure_ascii=False)[:6000]}

    Complete current best engine.py:
    ```python
    {best_code}
    ```

    Goal: produce a complete engine.py that improves the composite benchmark, especially decode and mixed throughput, with correctness as a hard gate. Start from the validated best code, inspect which priorities are still absent, and make any efficient refactor; do not reimplement optimizations already present. Keep the public API, dynamic config, request-state semantics, and generic correctness fallbacks. Shared packed KV remains a later, evidence-driven experiment. The tested baseline remains available if this candidate fails.

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
        General branch: improve the current best using benchmark feedback and the ordered
        optimization priorities below. Do not rely on token content beyond normal input handling.
        """).strip()

    token_profile = trace_summary.get("token_content_profile", {})
    return textwrap.dedent(f"""
    Token-aware experimental branch. The user has explicitly permitted sending this redacted
    token-content profile to the external LLM. Raw token ids are intentionally not included.

    Redacted token-content profile:
    {json.dumps(to_jsonable(token_profile), indent=2, ensure_ascii=False)[:4500]}

    Token-aware rules:
    - Use the profile only for grouping, sizing, or exact full-input hash caches; normal KV execution remains the fallback.
    - Any content branch must produce identical logits and preserve request order.
    - Never hard-code or log raw token ids, outputs, model dimensions, or request ids; never reuse stale logits without an exact full-history match.
    - Shared packed KV remains lower priority and requires measured KV overhead.
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
