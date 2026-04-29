"""
Conversation agent for NDA review.

Accepts (contract, user_prompt, history), retrieves supporting context from
the training corpus via the injected retriever, and calls the HuggingFace
Serverless Inference API (default: Qwen/Qwen2.5-7B-Instruct) to generate a
free-form answer.

Grounding rule enforced here (per Milestone 2 spec):
  Retrieved precedents may inform reasoning but ALL cited evidence must come
  verbatim from the Contract Under Review — never from retrieved examples.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from ..retrieval.base import BaseRetriever
from .history import ConversationHistory

# Default model — stays in the Qwen family used in Milestone 1.
# Any HF model that supports the chat_completion endpoint works here.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_SYSTEM_TEMPLATE = """\
You are an expert legal NDA analyst.  Your task is to answer questions about \
the NDA contract provided below.

══════════════════════════════════════════
CONTRACT UNDER REVIEW  (ID: {contract_id})
══════════════════════════════════════════
{contract_text}
══════════════════════════════════════════

EVIDENCE GROUNDING RULE (mandatory):
• Retrieved precedents supplied in user messages are reasoning context ONLY.
• Every piece of evidence you cite MUST be a verbatim quote from the Contract \
Under Review above.
• Mark every contract quote with the tag: [EVIDENCE: "exact quote"]
• Never fabricate, paraphrase, or cite retrieved precedents as contract evidence.

RESPONSE STRUCTURE:
1. Direct answer to the question
2. Contract evidence (using [EVIDENCE: "…"] tags)
3. Risk / implication note where relevant (concise)
"""

_MAX_PRECEDENTS = 5


class ConversationAgent:
    def __init__(
        self,
        retriever: BaseRetriever,
        hf_token: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self._client = InferenceClient(model=model, token=hf_token)

    # ── public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        contract: Dict[str, Any],
        user_prompt: str,
        history: ConversationHistory,
    ) -> Dict[str, Any]:
        """
        Process one conversation turn.

        Args:
            contract:    dict with at minimum {"text": str}.
                         Optional keys: "id" (str), "spans" (list).
            user_prompt: the user's current message.
            history:     ConversationHistory — updated in-place with this turn.

        Returns:
            {
                "response":       str,
                "evidence":       List[Dict],   # extracted + located in contract
                "precedents":     List[Dict],   # raw retrieval results
                "retrieval_mode": str,
                "usage":          {"prompt_tokens": int, "completion_tokens": int},
            }
        """
        contract_text = contract.get("text", "")
        contract_id = contract.get("id", "user-provided")

        # 1. Retrieve external context (training corpus precedents)
        precedents = self.retriever.retrieve(user_prompt, k=_MAX_PRECEDENTS)

        # 2. Build the full messages list for the chat API
        #    System message carries the contract (sent once, not repeated in history)
        system_content = _SYSTEM_TEMPLATE.format(
            contract_id=contract_id,
            contract_text=contract_text,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # 3. Replay prior turns
        messages.extend(history.to_anthropic_messages())

        # 4. Augment the current user message with retrieved precedents
        user_content = self._build_user_message(user_prompt, precedents)
        messages.append({"role": "user", "content": user_content})

        # 5. Call HuggingFace Inference API
        api_response = self._client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.4,
        )
        response_text: str = api_response.choices[0].message.content

        # 6. Extract evidence tags and locate them in the contract
        evidence = self._extract_evidence(response_text, contract_text)

        # 7. Persist the clean turn (user prompt only, not the augmented version)
        usage = api_response.usage
        history.add_turn(
            user_message=user_prompt,
            assistant_message=response_text,
            evidence=evidence,
            precedents=precedents,
            retrieval_mode=self.retriever.mode,
        )

        return {
            "response": response_text,
            "evidence": evidence,
            "precedents": precedents,
            "retrieval_mode": self.retriever.mode,
            "usage": {
                "prompt_tokens":     getattr(usage, "prompt_tokens",     0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            },
        }

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_user_message(user_prompt: str, precedents: List[Dict]) -> str:
        if not precedents:
            return user_prompt

        lines = []
        for i, p in enumerate(precedents, 1):
            label = p.get("label", "?")
            score = p.get("score", 0.0)
            lines.append(
                f"[Precedent {i} | {label} | similarity {score:.2f}]\n{p['text']}"
            )

        precedent_block = "\n\n".join(lines)
        return (
            "RETRIEVED PRECEDENTS — reasoning context only, do NOT cite as contract evidence:\n"
            "─────────────────────────────────────────────────────────────────────\n"
            f"{precedent_block}\n"
            "─────────────────────────────────────────────────────────────────────\n\n"
            f"QUESTION: {user_prompt}"
        )

    @staticmethod
    def _extract_evidence(
        response_text: str, contract_text: str
    ) -> List[Dict[str, Any]]:
        """
        Find all [EVIDENCE: "…"] markers and locate their character positions
        inside contract_text.  Returns deduplicated evidence list.
        """
        pattern = re.compile(r'\[EVIDENCE:\s*"((?:[^"\\]|\\.)*)"\]')
        seen: set[str] = set()
        evidence: List[Dict[str, Any]] = []

        for m in pattern.finditer(response_text):
            quote = m.group(1).strip()
            if quote in seen:
                continue
            seen.add(quote)

            # Exact match
            start = contract_text.find(quote)
            if start != -1:
                evidence.append(
                    {
                        "quote": quote,
                        "char_start": start,
                        "char_end": start + len(quote),
                        "verified": True,
                    }
                )
                continue

            # Case-insensitive fallback
            start_ci = contract_text.lower().find(quote.lower())
            if start_ci != -1:
                evidence.append(
                    {
                        "quote": quote,
                        "char_start": start_ci,
                        "char_end": start_ci + len(quote),
                        "verified": True,
                        "note": "case-normalised match",
                    }
                )
                continue

            evidence.append(
                {
                    "quote": quote,
                    "char_start": -1,
                    "char_end": -1,
                    "verified": False,
                    "note": "quote not found verbatim in contract",
                }
            )

        return evidence
