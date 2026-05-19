"""
Hypothesis Analyst agent — Member 2, Milestone 3.

Evaluates one NDA hypothesis at a time against a contract, returning a
structured verdict with label, confidence, evidence spans, and reasoning.

On retry (attempt > 1) the reviewer's feedback is injected into the prompt
so the model can correct specific issues.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient

from ..retrieval.base import BaseRetriever

DEFAULT_MODEL = "Qwen/Qwen3-8B"

_VALID_LABELS = {"ENTAILED", "CONTRADICTED", "NOT_MENTIONED"}

# ── prompts ───────────────────────────────────────────────────────────────────

_ANALYST_SYSTEM = """\
You are a legal Natural Language Inference (NLI) expert.
Given a contract text and a hypothesis, determine whether the hypothesis is
ENTAILED by, CONTRADICTED by, or NOT_MENTIONED in the contract.

Output ONLY valid JSON (no markdown fences, no extra text):
{
  "hypothesis_id": "<string>",
  "label": "<ENTAILED|CONTRADICTED|NOT_MENTIONED>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<explain why you chose this label and why the selected evidence supports it>",
  "evidence": [
    {
      "char_start": <int>,
      "char_end": <int>,
      "quote": "<exact substring of contract>",
      "relevance_score": <float 0.0-1.0>
    }
  ],
  "counter_evidence": ["<exact substring of contract that argues against your label>"]
}

Rules:
1. ENTAILED or CONTRADICTED -> at least one evidence span required.
2. NOT_MENTIONED -> evidence list may be empty.
3. char_start/char_end are 0-based character indices into the CONTRACT below.
4. quote must equal contract[char_start:char_end] exactly.
5. confidence is your certainty in the label (1.0 = certain, 0.0 = no evidence either way).
6. relevance_score per evidence span: how strongly does this quote support your label (1.0 = very strong, 0.0 = weak).
7. reasoning must explain the label decision and justify why the evidence quotes are relevant.
8. counter_evidence is an array of verbatim contract quotes that argue against your label; may be empty.\
"""

_CONTRACT_BLOCK = """\
══ CONTRACT (ID: {contract_id}) ══
{contract_text}
══ END CONTRACT ══"""

_PRECEDENTS_BLOCK = """\
── RETRIEVED PRECEDENTS (reasoning context only — do NOT cite as evidence) ──
{lines}
── END PRECEDENTS ──"""

_HYPOTHESIS_BLOCK = """\
Hypothesis {h_id} — {h_title}
"{h_text}"

Evaluate the contract above against this hypothesis and return ONLY the JSON object."""

_RETRY_PREFIX = """\
Your previous response was REJECTED by the reviewer.

Reviewer feedback:
{feedback}

Fix every issue described above. Return ONLY the corrected JSON object.

"""

_MAX_PRECEDENTS = 4
_MAX_TOKENS = 600


# ── main class ────────────────────────────────────────────────────────────────

class HypothesisAnalyst:
    """
    Evaluates one hypothesis per call against a contract.

    Args:
        retriever : initialised BaseRetriever (vector or graphrag).
        hf_token  : HuggingFace API token.
        model     : HF model string; must be from the Qwen3 family.
    """

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

    def analyze(
        self,
        contract: Dict[str, Any],
        hypothesis: Dict[str, Any],
        attempt: int = 1,
        reviewer_feedback: str = "",
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate one hypothesis against a contract.

        Args:
            contract:          dict with at minimum {"text": str, "id": str}.
            hypothesis:        dict with {"id": str, "text": str, "title": str}.
            attempt:           1-indexed retry count (1 = first attempt).
            reviewer_feedback: rejection feedback from ReviewerAgent (empty on attempt 1).

        Returns:
            (verdict_dict, tool_calls_list)

            verdict_dict keys:
                hypothesis_id, label, confidence, evidence_spans, reasoning

            tool_calls_list: runtrace-ready ToolCall dicts for retrieve + llm_call.
        """
        t0 = time.perf_counter()
        tool_calls: List[Dict[str, Any]] = []

        contract_text = contract.get("text", "")
        contract_id = contract.get("id", "unknown")
        h_id = hypothesis["id"]

        # 1. Retrieve precedents scoped to this hypothesis
        precedents = self.retriever.retrieve(
            query=hypothesis["text"],
            hypothesis_id=h_id,
            k=_MAX_PRECEDENTS,
            contract=contract,
        )
        tool_calls.append(self._retrieve_tool_call(h_id, hypothesis["text"], precedents))

        # 2. Build messages
        messages = self._build_messages(
            contract_id=contract_id,
            contract_text=contract_text,
            hypothesis=hypothesis,
            precedents=precedents,
            attempt=attempt,
            reviewer_feedback=reviewer_feedback,
        )

        # 3. Call the LLM
        raw_response = self._llm_call(messages)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Parse JSON from response
        verdict = self._parse_verdict(raw_response, h_id)

        # 5. Locate / verify evidence spans in contract text
        verdict["evidence"] = _locate_spans(
            verdict.get("evidence", []), contract_text
        )

        tool_calls.append(
            self._analyst_tool_call(h_id, attempt, raw_response, verdict, latency_ms)
        )

        return verdict, tool_calls

    # ── private: message building ─────────────────────────────────────────────

    def _build_messages(
        self,
        contract_id: str,
        contract_text: str,
        hypothesis: Dict[str, Any],
        precedents: List[Dict[str, Any]],
        attempt: int,
        reviewer_feedback: str,
    ) -> List[Dict[str, str]]:
        contract_block = _CONTRACT_BLOCK.format(
            contract_id=contract_id,
            contract_text=contract_text,
        )

        precedent_lines = "\n\n".join(
            f"[Precedent {i} | {p.get('label','?')} | score {p.get('score',0):.2f}]\n{p['text']}"
            for i, p in enumerate(precedents, 1)
        )
        precedents_block = (
            _PRECEDENTS_BLOCK.format(lines=precedent_lines)
            if precedents
            else ""
        )

        hypothesis_block = _HYPOTHESIS_BLOCK.format(
            h_id=hypothesis["id"],
            h_title=hypothesis.get("title", ""),
            h_text=hypothesis["text"],
        )

        user_parts = []
        if attempt > 1 and reviewer_feedback:
            user_parts.append(_RETRY_PREFIX.format(feedback=reviewer_feedback))
        user_parts.extend(filter(None, [contract_block, precedents_block, hypothesis_block]))

        return [
            {"role": "system", "content": _ANALYST_SYSTEM},
            {"role": "user",   "content": "\n\n".join(user_parts)},
        ]

    # ── private: LLM call ─────────────────────────────────────────────────────

    def _llm_call(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=_MAX_TOKENS,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"[HypothesisAnalyst] LLM call failed: {exc}")
            return ""

    # ── private: response parsing ─────────────────────────────────────────────

    def _parse_verdict(self, raw: str, h_id: str) -> Dict[str, Any]:
        fallback = {
            "hypothesis_id":  h_id,
            "label":          "NOT_MENTIONED",
            "confidence":     0.0,
            "reasoning":      "Parse error — could not extract valid JSON from model response.",
            "evidence":       [],
            "counter_evidence": [],
        }
        try:
            data = _extract_json(raw)
        except ValueError:
            return fallback

        # Normalise and validate
        label = str(data.get("label", "")).strip().upper()
        if label not in _VALID_LABELS:
            label = "NOT_MENTIONED"

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", "")).strip()

        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        counter_evidence = data.get("counter_evidence", [])
        if not isinstance(counter_evidence, list):
            counter_evidence = []
        # keep only plain strings; drop any accidental objects
        counter_evidence = [c for c in counter_evidence if isinstance(c, str)]

        return {
            "hypothesis_id":   h_id,
            "label":           label,
            "confidence":      confidence,
            "reasoning":       reasoning,
            "evidence":        evidence,
            "counter_evidence": counter_evidence,
        }

    # ── private: runtrace tool call builders ──────────────────────────────────

    @staticmethod
    def _retrieve_tool_call(
        h_id: str, query: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "name": "retrieve",
            "args": {"hypothesis_id": h_id, "query": query[:120]},
            "output": {"count": len(results), "top_score": results[0]["score"] if results else None},
            "count": len(results),
        }

    @staticmethod
    def _analyst_tool_call(
        h_id: str,
        attempt: int,
        raw_response: str,
        verdict: Dict[str, Any],
        latency_ms: float,
    ) -> Dict[str, Any]:
        return {
            "name": "hypothesis_analyst",
            "args": {"hypothesis_id": h_id, "attempt": attempt},
            "output": {
                "label":                 verdict.get("label"),
                "confidence":            verdict.get("confidence"),
                "reasoning":             verdict.get("reasoning", ""),
                "evidence_count":        len(verdict.get("evidence", [])),
                "counter_evidence_count": len(verdict.get("counter_evidence", [])),
                "raw_response_len":      len(raw_response),
            },
            "count":      attempt,
            "latency_ms": latency_ms,
        }


# ── module-level helpers ──────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from an LLM response."""
    # Strip Qwen3 thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Extract first {...} block
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in: {text[:300]!r}")


def _locate_spans(
    evidence: List[Dict[str, Any]], contract_text: str
) -> List[Dict[str, Any]]:
    """
    Verify / correct char_start and char_end for each evidence span.

    The model provides indices but they are often off.  We re-search the contract
    so that rule 4 (quote == contract[char_start:char_end]) always holds.
    relevance_score and any other model-provided fields are preserved.
    """
    located: List[Dict[str, Any]] = []
    for span in evidence:
        quote = span.get("quote", "").strip()
        if not quote:
            continue

        start = contract_text.find(quote)
        if start != -1:
            located.append({
                **span,
                "char_start": start,
                "char_end":   start + len(quote),
            })
            continue

        # Case-insensitive fallback
        start_ci = contract_text.lower().find(quote.lower())
        if start_ci != -1:
            located.append({
                **span,
                "char_start": start_ci,
                "char_end":   start_ci + len(quote),
                "note":       "case-normalised match",
            })
            continue

        located.append({**span, "char_start": -1, "char_end": -1, "note": "not found verbatim"})

    return located
