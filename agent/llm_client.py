from __future__ import annotations

import json
import os
import re
import textwrap
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
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))
        self.enabled = bool(self.api_key and self.model and os.getenv("AGENT_USE_LLM", "1") != "0")

    def ask_for_candidate(self, prompt: str) -> dict[str, str] | None:
        if not self.enabled:
            self.log.log("llm", "LLM client disabled or missing API_KEY/BASE_MODEL")
            return None
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert LLM inference runtime engineer. Return only the requested fixed-format answer.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 6000,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = obj["choices"][0]["message"]["content"]
            parsed = parse_llm_response(content)
            self.log.log("llm", "received LLM optimization proposal", {
                "strategy": parsed.get("strategy", ""),
                "expected_benefit": parsed.get("expected_benefit", ""),
                "risk": parsed.get("risk", ""),
                "self_check_notes": parsed.get("self_check_notes", ""),
            })
            return parsed
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            self.log.log("llm", "LLM request failed; continuing with deterministic templates", {"error": repr(exc)})
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
    results: list[CandidateResult],
    best_code_excerpt: str,
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
    return textwrap.dedent(f"""
    Current model/environment summary:
    {json.dumps(to_jsonable(env_summary), indent=2, ensure_ascii=False)[:5000]}

    Observed trace/benchmark summary:
    {json.dumps(to_jsonable(trace_summary), indent=2, ensure_ascii=False)[:4000]}

    Runtime hard constraints:
    {json.dumps(spec, indent=2, ensure_ascii=False)}

    Candidate results so far:
    {json.dumps(to_jsonable(candidate_summary), indent=2, ensure_ascii=False)[:6000]}

    Current best engine excerpt:
    ```python
    {best_code_excerpt[:5000]}
    ```

    Goal: first preserve correctness, then improve decode/mixed throughput. Keep the public interface unchanged, read config dynamically, preserve request state semantics, and avoid hard-coded model dimensions.

    Return exactly this format. If you provide code, provide a full engine.py implementation inside patch_or_full_engine, not a diff:
    strategy:
    expected_benefit:
    risk:
    patch_or_full_engine:
    self_check_notes:
    """).strip()
