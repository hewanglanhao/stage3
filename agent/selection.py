from __future__ import annotations

import shutil

from common import AgentLog, CandidateResult, WORKSPACE


def pick_best(results: list[CandidateResult], log: AgentLog) -> CandidateResult:
    passing = [result for result in results if result.correctness_ok and result.stress_ok]
    if passing:
        best = max(passing, key=lambda result: (result.benchmark_ok, result.score, -result.iteration))
        log.log("selection", "selected best passing candidate", {
            "iteration": best.iteration,
            "name": best.name,
            "score": best.score,
            "benchmark_ok": best.benchmark_ok,
        })
        return best
    prechecked = [result for result in results if result.precheck_ok]
    if prechecked:
        best = prechecked[0]
        log.log("selection", "no candidate passed correctness; falling back to first prechecked candidate", {"name": best.name})
        return best
    raise RuntimeError("no candidate could be imported")


def install_best(best: CandidateResult, log: AgentLog) -> None:
    destination = WORKSPACE / "engine.py"
    shutil.copy2(best.path, destination)
    log.log("selection", "installed final engine.py", {"source": best.path, "destination": destination})
