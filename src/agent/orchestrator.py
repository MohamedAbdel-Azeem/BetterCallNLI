"""
Orchestrator for BetterCallNLI — Milestone 3.

Top-level coordinator that replaces the direct `ConversationAgent.chat()` calls
in app.py and cli.py.  Every entry point (Streamlit UI, CLI) should instantiate
one Orchestrator and call `orchestrator.run(contract, user_message, history)`.

Flow
----
    user_message
        │
        ▼
    IntentRouter.route()
        │
        ├─ "conversation"         ──► ConversationAgent.chat()
        │                                  └─ returns ConversationResult
        │
        └─ "hypothesis_analysis"  ──► HypothesisPipeline.run()
                                           └─ returns HypothesisResult

Both branches return a unified result dict that callers can handle uniformly.

Runtrace
--------
Every call to `run()` produces a `tool_calls` list that starts with the
IntentRouter ToolCall entry and is extended by the downstream agent.
The Orchestrator itself does NOT write runtraces to disk — that is the
responsibility of RuntraceFormatter (Task 4).  It does, however, attach
the `tool_calls` list to the result dict so Task 4 has everything it needs.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..enrichment import PlaybookEnricher
from ..retrieval.base import BaseRetriever
from ..utils.runtrace import RuntraceFormatter
from .conversation_agent import ConversationAgent
from .history import ConversationHistory
from .hypothesis_pipeline import HypothesisPipeline  # stub until Tasks 2+3
from .intent_router import IntentRouter
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, formatter-compatible."""
    return datetime.now(timezone.utc).isoformat()

# ── default model (matches the rest of the codebase) ─────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class Orchestrator:
    """
    Single entry point for all user interactions.

    Args:
        retriever     : an initialised BaseRetriever (Vector or GraphRAG).
        hf_token      : HuggingFace API token.
        playbook_path : path to playbook.yaml — forwarded to HypothesisPipeline.
        model         : HF model string shared by all agents.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        hf_token: str,
        playbook_path: str = "playbook.yaml",
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.retriever = retriever
        self.hf_token = hf_token
        self.playbook_path = playbook_path
        self.model = model

        # ── sub-components ────────────────────────────────────────────────────
        self.router = IntentRouter(hf_token=hf_token, model="Qwen/Qwen2.5-0.5B-Instruct")
        self.conv_agent = ConversationAgent(
            retriever=retriever,
            hf_token=hf_token,
            model=model,
        )

        self.hypothesis_pipeline = HypothesisPipeline(
            retriever=retriever,
            hf_token=hf_token,
            playbook_path=playbook_path,
        )

        # ── Task 4 wiring: playbook enrichment + runtrace formatter ───────────
        # Additive only — these are consulted AFTER the pipeline runs to apply
        # the (unmodified) playbook and emit a schema-compliant runtrace dict
        # on every contract-mode call. Conversation-mode session runtraces are
        # built on demand via build_session_runtrace().
        self.enricher  = PlaybookEnricher(playbook_path=playbook_path)
        self.formatter = RuntraceFormatter(playbook_path=playbook_path, model=model)

        # Per-turn router calls accumulated for conversation-mode sessions.
        # The caller (CLI) is free to reset this between sessions.
        self._session_router_calls: List[Dict[str, Any]] = []
        self._session_started_at: Optional[str] = None

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        contract: Dict[str, Any],
        user_message: str,
        history: ConversationHistory,
        hypotheses: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Process one user interaction.

        Args:
            contract:     dict with at minimum {"text": str, "id": str}.
            user_message: the raw string the user typed.
            history:      ConversationHistory for the current session.
                          Updated in-place by ConversationAgent when in
                          conversation mode; unchanged in hypothesis mode.

        Returns:
            Unified result dict.  Always contains:
                "mode"       : "conversation" | "hypothesis_analysis"
                "tool_calls" : List[Dict]   — runtrace-ready ToolCall entries,
                                              starting with the router call.

            Conversation mode additionally contains:
                "response"       : str
                "evidence"       : List[Dict]
                "precedents"     : List[Dict]
                "retrieval_mode" : str
                "usage"          : Dict

            Hypothesis mode additionally contains:
                "verdicts"      : List[Dict]   (H01–H17)
                "agent_traces"  : List[Dict]
        """
        started_at = _utc_now_iso()
        if self._session_started_at is None:
            self._session_started_at = started_at

        # 1. Route the message
        intent, router_tool_call = self.router.route(user_message)
        tool_calls: List[Dict] = [router_tool_call]

        # Track this router call for session-runtrace assembly later.
        self._session_router_calls.append(router_tool_call)

        # 2. Dispatch to the appropriate agent
        if intent == "conversation":
            result = self._run_conversation(contract, user_message, history)
        else:
            result = self._run_hypothesis_analysis(contract, hypotheses=hypotheses)

        # 3. Attach routing metadata to the result (existing behaviour)
        result["mode"] = intent
        result["tool_calls"] = tool_calls + result.pop("_agent_tool_calls", [])

        # 4. Task 4 wiring — enrich + emit runtrace for hypothesis-mode calls.
        #    Conversation mode runtraces are built on demand at session end via
        #    build_session_runtrace(); they need the full turn history.
        ended_at = _utc_now_iso()
        if intent == "hypothesis_analysis":
            try:
                verdicts = result.get("verdicts", [])
                enriched = self.enricher.enrich(verdicts)
                result["enriched_verdicts"] = enriched
                result["runtrace"] = self.formatter.build_contract_runtrace(
                    contract           = contract,
                    intent_router_call = router_tool_call,
                    agent_traces       = result.get("agent_traces", []),
                    enriched_verdicts  = enriched,
                    retrieval_mode     = getattr(self.retriever, "mode", "graphrag"),
                    started_at         = started_at,
                    ended_at           = ended_at,
                    model              = self.model,
                )
            except Exception as exc:                                            # pragma: no cover
                # Never let runtrace/enrichment failure break the pipeline.
                # Surface the error in the result dict so the caller can react.
                result["runtrace_error"] = f"{type(exc).__name__}: {exc}"

        return result

    # ── private dispatch methods ───────────────────────────────────────────────

    def _run_conversation(
        self,
        contract: Dict[str, Any],
        user_message: str,
        history: ConversationHistory,
    ) -> Dict[str, Any]:
        """
        Delegate to ConversationAgent and normalise the result shape.
        ConversationAgent.chat() already updates `history` in-place.
        """
        agent_result = self.conv_agent.chat(
            contract=contract,
            user_prompt=user_message,
            history=history,
        )
        # Carry forward any tool_calls the agent recorded (none in MS2, but
        # Task 4 / RuntraceFormatter will expect the key to exist)
        agent_result["_agent_tool_calls"] = agent_result.pop("tool_calls", [])
        return agent_result

    def _run_hypothesis_analysis(
        self,
        contract: Dict[str, Any],
        hypotheses: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Delegate to HypothesisPipeline.
        Currently raises NotImplementedError (stub) until Tasks 2 & 3 land.
        """
        pipeline_result = self.hypothesis_pipeline.run(contract, hypotheses=hypotheses)
        pipeline_result["_agent_tool_calls"] = pipeline_result.pop("tool_calls", [])
        return pipeline_result

    # ── conversation-session runtrace helpers (Task 4) ────────────────────────

    def build_session_runtrace(
        self,
        contract: Dict[str, Any],
        history:  ConversationHistory,
        ended_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build one schema-compliant conversation-mode runtrace covering the
        entire current session. Intended to be called by the CLI (or any
        front-end) when the user ends the conversation.

        Reads turns directly from `history` and pairs them with the
        intent-router calls accumulated by `run()` during the session.
        ConversationAgent tool_calls are reconstructed by the formatter
        from the per-turn `evidence`/`precedents`/`assistant` data already
        stored in history — no changes to ConversationAgent or History.

        Args:
            contract : the contract dict the session is anchored to.
            history  : the ConversationHistory tracked across all turns.
            ended_at : optional ISO-8601 UTC end timestamp. Defaults to now.

        Returns:
            A schema-compliant MS3 conversation-mode runtrace dict.
        """
        started_at = self._session_started_at or _utc_now_iso()
        ended_at   = ended_at or _utc_now_iso()

        turn_records: List[Dict[str, Any]] = []
        for idx, turn in enumerate(history.turns, start=1):
            router_call = (
                self._session_router_calls[idx - 1]
                if idx - 1 < len(self._session_router_calls)
                else {}
            )
            turn_records.append({
                "turn_id":            turn.get("turn_id", idx),
                "timestamp":          turn.get("timestamp"),
                "user_prompt":        turn.get("user", ""),
                "assistant_response": turn.get("assistant", ""),
                "evidence":           turn.get("evidence", []) or [],
                "precedents":         turn.get("precedents", []) or [],
                "retrieval_mode":     turn.get("retrieval_mode", getattr(self.retriever, "mode", "graphrag")),
                "intent_router_call": router_call,
            })

        return self.formatter.build_conversation_runtrace(
            contract        = contract,
            session_id      = history.session_id,
            turn_records    = turn_records,
            retrieval_mode  = getattr(self.retriever, "mode", "graphrag"),
            started_at      = started_at,
            ended_at        = ended_at,
            model           = self.model,
            created_at      = getattr(history, "created_at", None),
        )

    def reset_session(self) -> None:
        """Clear accumulated per-turn router calls. Call between sessions."""
        self._session_router_calls = []
        self._session_started_at   = None


# ── convenience factory ────────────────────────────────────────────────────────

def build_orchestrator(
    retrieval_mode: str = "graphrag",
    playbook_path: str = "playbook.yaml",
    model: str = DEFAULT_MODEL,
) -> Orchestrator:
    """
    Factory used by both app.py and cli.py to build a fully wired Orchestrator
    from environment variables.

    Args:
        retrieval_mode : "graphrag" or "vector"
        playbook_path  : path to playbook.yaml
        model          : HF model string

    Returns:
        A ready Orchestrator instance.

    Raises:
        EnvironmentError : if HF_TOKEN is not set.
        RuntimeError     : if the requested retriever fails to connect.
    """
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is not set. Add it to your .env file.")

    retriever = _build_retriever(retrieval_mode)

    return Orchestrator(
        retriever=retriever,
        hf_token=hf_token,
        playbook_path=playbook_path,
        model=model,
    )


def _build_retriever(mode: str) -> BaseRetriever:
    if mode == "vector":
        from ..retrieval.vector_rag import VectorRAGRetriever
        r = VectorRAGRetriever()
        if not r.is_ready():
            raise RuntimeError(
                "VectorRAGRetriever could not connect. "
                "Check CHROMA_API_KEY in your .env file."
            )
        return r

    # default: graphrag
    from ..retrieval.graphrag_retriever import GraphRAGRetriever
    uri  = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "")
    pwd  = os.getenv("NEO4J_PASSWORD", "")
    if not all([uri, user, pwd]):
        raise RuntimeError(
            "GraphRAGRetriever requires NEO4J_URI, NEO4J_USERNAME, "
            "and NEO4J_PASSWORD in your .env file."
        )
    r = GraphRAGRetriever(uri=uri, username=user, password=pwd)
    if not r.connect():
        raise RuntimeError(
            "GraphRAGRetriever failed to connect to Neo4j. "
            "Check your NEO4J_* credentials."
        )
    return r