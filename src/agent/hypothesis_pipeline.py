"""
Hypothesis Pipeline — Member 2, Milestone 3.

Loops over all 17 hypotheses from the playbook and runs:
    HypothesisAnalyst → ReviewerAgent (max 3 retries per hypothesis)

Returns a unified result dict consumed by the Orchestrator and eventually
the RuntraceFormatter (Task 4).

ReviewerAgent is imported with a graceful fallback stub so this file
works immediately, even before Member 3 delivers reviewer_agent.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..retrieval.base import BaseRetriever
from .hypothesis_analyst import HypothesisAnalyst

DEFAULT_MODEL = "Qwen/Qwen3-8B"

# ── ReviewerAgent — real or stub ──────────────────────────────────────────────

try:
    from .reviewer_agent import ReviewerAgent as _ReviewerAgent

    class ReviewerAgent(_ReviewerAgent):  # type: ignore[misc]
        pass

except ImportError:
    class ReviewerAgent:  # type: ignore[no-redef]
        """Stub — auto-replaced when src/agent/reviewer_agent.py is ready."""

        def __init__(self, hf_token: str, model: str = DEFAULT_MODEL) -> None:
            pass

        def review(
            self,
            verdict: Dict[str, Any],
            contract: Dict[str, Any],
            hypothesis: Dict[str, Any],
        ) -> Dict[str, Any]:
            return {"accepted": True, "rejection_reasons": [], "feedback": ""}


# ── playbook loader ───────────────────────────────────────────────────────────

def _load_playbook(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Playbook not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── pipeline ──────────────────────────────────────────────────────────────────

class HypothesisPipeline:
    """
    Orchestrates the per-hypothesis Analyst → Reviewer loop for all 17 hypotheses.

    Args:
        retriever     : initialised BaseRetriever (vector or graphrag).
        hf_token      : HuggingFace API token.
        playbook_path : path to playbook.yaml.
        model         : HF model string (Qwen3 family).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        hf_token: str,
        playbook_path: str = "playbook.yaml",
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.analyst  = HypothesisAnalyst(retriever, hf_token, model=model)
        self.reviewer = ReviewerAgent(hf_token=hf_token, model=model)
        self.playbook = _load_playbook(playbook_path)

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse a contract against all 17 playbook hypotheses.

        Args:
            contract: dict with at minimum {"text": str, "id": str}.

        Returns:
            {
                "verdicts":     List[Dict]  — one verdict dict per hypothesis (H01–H17)
                "agent_traces": List[Dict]  — one trace per hypothesis for runtrace
                "tool_calls":   List[Dict]  — all tool calls across the full run
            }
        """
        verdicts:     List[Dict[str, Any]] = []
        agent_traces: List[Dict[str, Any]] = []
        all_tool_calls: List[Dict[str, Any]] = []

        for check in self.playbook["checks"]:
            hyp = {
                "id":    check["hypothesis_id"],
                "text":  check["hypothesis_text"],
                "title": check.get("title", ""),
            }

            verdict:          Dict[str, Any] | None = None
            candidate:        Dict[str, Any] = {}
            feedback:         str = ""
            trace_tool_calls: List[Dict[str, Any]] = []
            attempts:         int = 0
            accepted:         bool = False

            for attempt in range(1, 4):
                attempts = attempt

                # ── Analyst ──────────────────────────────────────────────────
                candidate, analyst_tcs = self.analyst.analyze(
                    contract=contract,
                    hypothesis=hyp,
                    attempt=attempt,
                    reviewer_feedback=feedback,
                )
                trace_tool_calls.extend(analyst_tcs)

                # ── Reviewer ─────────────────────────────────────────────────
                rev_result = self.reviewer.review(candidate, contract, hyp)

                # Collect reviewer tool calls if provided (Member 3 may add them)
                rev_tcs = rev_result.pop("_tool_calls", [])
                trace_tool_calls.extend(rev_tcs)

                if rev_result.get("accepted", False):
                    verdict   = candidate
                    accepted  = True
                    break

                feedback = rev_result.get("feedback", "")

            # Best-effort: use last candidate if all 3 attempts rejected
            if verdict is None:
                verdict = candidate

            verdicts.append(verdict)
            all_tool_calls.extend(trace_tool_calls)
            agent_traces.append({
                "agent_id":      "hypothesis_analyst",
                "hypothesis_id": hyp["id"],
                "attempts":      attempts,
                "tool_calls":    trace_tool_calls,
                "final_verdict": verdict,
                "accepted":      accepted,
            })

        return {
            "verdicts":     verdicts,
            "agent_traces": agent_traces,
            "tool_calls":   all_tool_calls,
        }
