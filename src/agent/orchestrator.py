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
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from ..retrieval.base import BaseRetriever
from .conversation_agent import ConversationAgent
from .history import ConversationHistory
from .hypothesis_pipeline import HypothesisPipeline  # stub until Tasks 2+3
from .intent_router import IntentRouter

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
        self.router = IntentRouter(hf_token=hf_token, model=model)

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

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        contract: Dict[str, Any],
        user_message: str,
        history: ConversationHistory,
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
        # 1. Route the message
        intent, router_tool_call = self.router.route(user_message)
        tool_calls: List[Dict] = [router_tool_call]

        # 2. Dispatch to the appropriate agent
        if intent == "conversation":
            result = self._run_conversation(contract, user_message, history)
        else:
            result = self._run_hypothesis_analysis(contract)

        # 3. Attach routing metadata to the result
        result["mode"] = intent
        result["tool_calls"] = tool_calls + result.pop("_agent_tool_calls", [])

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
    ) -> Dict[str, Any]:
        """
        Delegate to HypothesisPipeline.
        Currently raises NotImplementedError (stub) until Tasks 2 & 3 land.
        """
        pipeline_result = self.hypothesis_pipeline.run(contract)
        pipeline_result["_agent_tool_calls"] = pipeline_result.pop("tool_calls", [])
        return pipeline_result


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