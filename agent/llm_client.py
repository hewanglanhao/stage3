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
        content = self._chat_completion(
            system_prompt="You are an expert LLM inference runtime engineer. Return only the requested fixed-format answer.",
            user_prompt=prompt,
            max_tokens=6000,
            temperature=0.2,
            purpose="engine candidate",
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
            return obj["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            self.log.log("llm", f"LLM request failed for {purpose}; continuing with fallback", {"error": repr(exc)})
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
    results: list[CandidateResult],
    best_code_excerpt: str,
    feedback: dict[str, Any],
    llm_round: int,
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
    LLM optimization round: {llm_round}

    Current model/environment summary:
    {json.dumps(to_jsonable(env_summary), indent=2, ensure_ascii=False)[:5000]}

    Observed trace/benchmark summary:
    {json.dumps(to_jsonable(trace_summary), indent=2, ensure_ascii=False)[:4000]}

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

    Goal: use the defects/guidance above to produce the next full engine.py. First preserve correctness, then improve decode/mixed throughput. Keep the public interface unchanged, read config dynamically, preserve request state semantics, and avoid hard-coded model dimensions.

    Return exactly this format. Provide a full engine.py implementation inside patch_or_full_engine, not a diff:
    strategy:
    expected_benefit:
    risk:
    patch_or_full_engine:
    self_check_notes:
    """).strip()
