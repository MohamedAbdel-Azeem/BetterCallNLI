"""
Intent Router for BetterCallNLI.

Classifies a user message into one of two modes before any downstream
pipeline runs:

    "conversation"        — user is asking a general question about the contract
    "hypothesis_analysis" — user wants a full 17-hypothesis structured review

Routing is done via a single lightweight LLM call.  The response is
normalised and validated; any ambiguous or malformed output falls back
to "conversation" (safe default — never kicks off an expensive pipeline
unintentionally).

Runtrace:
    Every call records a ToolCall entry:
        name   : "intent_router"
        args   : {"user_message": str}
        output : {"intent": str, "raw_response": str, "fallback_used": bool}
        latency_ms: float
"""

from __future__ import annotations

import os
import time
from typing import Dict, Literal, Tuple

from huggingface_hub import InferenceClient

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

IntentType = Literal["conversation", "hypothesis_analysis"]

# Keywords that strongly signal the user wants a full hypothesis run.
# Used as a deterministic pre-check before spending an LLM call.
_HYPOTHESIS_TRIGGERS: list[str] = [
    "analyze", "analyse", "analysis",
    "full review", "run review", "full analysis",
    "all hypotheses", "all hypothesis", "check hypotheses",
    "hypothesis analysis", "17 hypotheses", "evaluate contract",
    "nli", "contract review", "structured review",
    "run pipeline", "generate report",
]

_SYSTEM_PROMPT = """\
You are a routing assistant for a legal NDA review system.

Given a user message, classify its intent as exactly one of:
  - conversation        : the user is asking a general question about a contract
                          (e.g. "what are the confidentiality obligations?",
                          "does this NDA allow sharing with employees?",
                          "summarise the termination clause")
  - hypothesis_analysis : the user wants a full structured NDA analysis against
                          the 17 standard hypotheses from ContractNLI
                          (e.g. "analyze this contract", "run full review",
                          "check all hypotheses", "generate the NDA report")

Rules:
  1. Respond with ONLY one of the two exact strings above.
  2. No punctuation, no explanation, no extra words.
  3. When uncertain, prefer: conversation
"""

_USER_TEMPLATE = "User message: {message}"


# ── main class ────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Lightweight router that decides whether a user message should be handled
    by the ConversationAgent or the HypothesisPipeline.

    Args:
        hf_token : HuggingFace API token.
        model    : HF Inference endpoint model string.
    """

    def __init__(
        self,
        hf_token: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = InferenceClient(
            provider=os.getenv("HF_PROVIDER", "featherless-ai"),
            model=model,
            token=hf_token,
        )
        self.model = model

    # ── public API ────────────────────────────────────────────────────────────

    def route(self, user_message: str) -> Tuple[IntentType, Dict]:
        """
        Classify the user message and return the intent plus a runtrace
        ToolCall dict.

        Args:
            user_message: the raw string the user typed.

        Returns:
            (intent, tool_call_dict)

            intent        : "conversation" | "hypothesis_analysis"
            tool_call_dict: runtrace-ready dict with name/args/output/latency_ms
        """
        t0 = time.perf_counter()

        # ── fast deterministic pre-check ──────────────────────────────────────
        deterministic_intent = self._keyword_check(user_message)
        if deterministic_intent is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            tool_call = self._build_tool_call(
                user_message=user_message,
                intent=deterministic_intent,
                raw_response="[keyword match — no LLM call]",
                fallback_used=False,
                latency_ms=latency_ms,
            )
            return deterministic_intent, tool_call

        # ── LLM classification ────────────────────────────────────────────────
        raw, fallback_used = self._llm_classify(user_message)
        intent = self._parse_response(raw)
        latency_ms = (time.perf_counter() - t0) * 1000

        tool_call = self._build_tool_call(
            user_message=user_message,
            intent=intent,
            raw_response=raw,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
        )
        return intent, tool_call

    # ── private helpers ───────────────────────────────────────────────────────

    def _keyword_check(self, message: str) -> IntentType | None:
        """
        Fast path: if the message contains a strong hypothesis-trigger keyword,
        skip the LLM call and route directly to hypothesis_analysis.
        Returns None if no deterministic match.
        """
        lowered = message.lower()
        for trigger in _HYPOTHESIS_TRIGGERS:
            if trigger in lowered:
                return "hypothesis_analysis"
        return None

    def _llm_classify(self, user_message: str) -> Tuple[str, bool]:
        """
        Call the LLM and return (raw_response_text, fallback_used).
        fallback_used=True means the LLM call failed and we used the default.
        """
        try:
            response = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _USER_TEMPLATE.format(message=user_message)},
                ],
                max_tokens=10,      # we only need one word
                temperature=0.0,    # deterministic
            )
            raw: str = response.choices[0].message.content.strip()
            return raw, False
        except Exception as exc:
            print(f"[IntentRouter] LLM call failed ({exc}), defaulting to 'conversation'")
            return "conversation", True

    @staticmethod
    def _parse_response(raw: str) -> IntentType:
        """
        Normalise the LLM output to one of the two valid intent strings.
        Falls back to "conversation" for anything unrecognised.
        """
        normalised = raw.strip().lower().rstrip(".")

        if normalised == "hypothesis_analysis":
            return "hypothesis_analysis"
        if normalised == "conversation":
            return "conversation"

        # Partial-match fallback (model sometimes adds extra words)
        if "hypothesis" in normalised or "analysis" in normalised:
            return "hypothesis_analysis"

        # Safe default
        return "conversation"

    @staticmethod
    def _build_tool_call(
        user_message: str,
        intent: IntentType,
        raw_response: str,
        fallback_used: bool,
        latency_ms: float,
    ) -> Dict:
        return {
            "name": "intent_router",
            "args": {
                "user_message": user_message,
            },
            "output": {
                "intent":        intent,
                "raw_response":  raw_response,
                "fallback_used": fallback_used,
            },
            "latency_ms": round(latency_ms, 2),
        }