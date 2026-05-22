"""
Reviewer Agent — judges HypothesisAnalyst output without seeing the full contract.

Receives hypothesis, predicted label, evidence quotes, and reasoning.
Returns a score (1–10) and acceptance decision based on a configurable threshold.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from huggingface_hub import InferenceClient

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_REVIEWER_SYSTEM = """\
You are a Legal Reasoning Judge in a Contract NLI (Natural Language Inference) pipeline.

Your role is to evaluate the output of an Analyst model that classified a legal contract
clause against a given hypothesis. You do NOT receive the full contract — the Analyst has
already extracted the relevant evidence. You validate what you are given.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU RECEIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Hypothesis      — The statement being evaluated against the contract
2. Label           — The Analyst's classification: entailed | contradicted | not_mentioned
3. Evidence        — Excerpt(s) from the contract (only present for entailed/contradicted)
4. Reasoning       — The Analyst's explanation of why the label fits the evidence


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR EVALUATION DIMENSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluate the Analyst's output across three dimensions:

── 1. LABEL ALIGNMENT ──────────────────────────────────
Does the label correctly reflect the relationship between the hypothesis and the evidence?

  • entailed     → The evidence must clearly and directly CONFIRM the hypothesis.
                   A partial or implied match is NOT sufficient.

  • contradicted → The evidence must clearly and directly CONFLICT with the hypothesis.
                   A tangential or weakly opposing excerpt is NOT sufficient.

  • not_mentioned → No evidence should be present. If evidence IS provided
                    alongside this label, that is an automatic score of 1.

Score this dimension:
  strong   → Label is unambiguously correct given the evidence
  debatable → Label is plausible but another label could also reasonably apply
  wrong    → Label is clearly incorrect given the evidence


── 2. EVIDENCE QUALITY ─────────────────────────────────
Is the evidence a legitimate, specific, and relevant contract excerpt?

  • For entailed / contradicted: evidence MUST be present.
    If missing → automatic score of 1.

  • The evidence must directly address the subject of the hypothesis.
    Generic or loosely related clauses are not valid evidence.

  • The evidence must read like a real contract excerpt — specific, formal, precise.
    Vague summaries or paraphrases are not valid evidence.

  • If multiple evidence excerpts are provided, ALL must be valid.
    One invalid excerpt fails this dimension.

  • For not_mentioned: evidence must be ABSENT.

Score this dimension:
  strong   → Evidence is specific, clearly relevant, and directly does the logical work
  weak     → Evidence is present but vague, tangential, or only partially relevant
  absent   → Evidence is missing when required, or present when it should not be
  not_required → Label is not_mentioned and no evidence was provided (correct)


── 3. REASONING COHERENCE ──────────────────────────────
Does the Analyst's reasoning correctly connect the evidence to the label?

  • The reasoning must not introduce facts, assumptions, or legal context
    not present in the evidence itself.

  • The reasoning must not overstate what the evidence says.

  • The logical chain from evidence → label must be explicit and sound.

  • For not_mentioned: the reasoning should explain why no relevant clause was found,
    not speculate about what the contract might say.

Score this dimension:
  clear  → Reasoning is logical, grounded in the evidence, and explains the label well
  loose  → Reasoning is directionally correct but makes unsupported leaps
  flawed → Reasoning contradicts itself, misreads the evidence, or is not grounded


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After evaluating all three dimensions, assign an overall score from 1 to 10.

Use this as a guide:

  9 - 10  → All three dimensions are strong. Output is reliable and well-supported.
  7 - 8   → Mostly strong with minor weaknesses. Label is correct but reasoning
             or evidence could be sharper.
  5 - 6   → Noticeable issues in one or two dimensions. Label may be correct but
             the support is weak or the reasoning makes unsupported assumptions.
  3 - 4   → Significant issues. The label is debatable or the evidence does not
             clearly do the required logical work.
  1 - 2   → Fundamental failure. Wrong label, missing required evidence, evidence
             present for not_mentioned, or completely incoherent reasoning.

The score reflects overall output quality — not a mechanical average of dimensions.
A single critical failure (e.g., missing evidence for entailed) should anchor the
score low regardless of how well the other dimensions perform.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU MUST NOT DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗ Do not attempt to recall or infer anything about the full contract
  ✗ Do not apply general legal knowledge to decide what the contract "should" say
  ✗ Do not reward a plausible-sounding output that lacks proper evidence
  ✗ Do not penalize the Analyst for stylistic choices or phrasing preferences
  ✗ Do not approve an output because it seems directionally right — apply the criteria strictly
  ✗ Do not add commentary, explanations, or any text outside the required JSON block


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond ONLY with the following JSON. No preamble. No explanation outside the JSON.

{
  "score": <integer from 1 to 10>,
  "flags": {
    "label_alignment":       "strong"       | "debatable" | "wrong",
    "evidence_quality":      "strong"       | "weak"      | "absent" | "not_required",
    "reasoning_coherence":   "clear"        | "loose"     | "flawed"
  },
  "critique": "<what failed and exactly what the Analyst must correct on the next attempt.>"
}

The "critique" field on a low score is critical — the Analyst will receive it directly
as feedback and must use it to produce a better output. Be specific about which dimension
failed and what needs to change. Do not be vague.\
"""

_USER_TEMPLATE = """\
Hypothesis: {hypothesis}

Label: {label}

Evidence:
{evidence}

Reasoning: {reasoning}\
"""

_MAX_TOKENS = 400

# ── main class ────────────────────────────────────────────────────────────────


class ReviewerAgent:
    """
    Judges Hypothesis Analyst output against a scored rubric.

    Args:
        hf_token  : HuggingFace API token.
        model     : HF model string (default: Qwen/Qwen2.5-7B-Instruct).
        threshold : Minimum score (1-10) to accept an analyst verdict.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        threshold: int = 5,
    ) -> None:
        self.threshold = threshold
        self.model = model
        token = os.getenv("HF_TOKEN")
        self._client = InferenceClient(
            provider=os.getenv("HF_PROVIDER", "featherless-ai"),
            model=model,
            token=token,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def review(
        self,
        verdict: Dict[str, Any],
        hypothesis: Dict[str, Any],
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate analyst verdict without access to the full contract.

        Args:
            verdict:    Analyst output dict (label, evidence, reasoning, hypothesis_id).
            hypothesis: Dict with keys id, text, title.
            attempt:    1-indexed attempt number (used for runtrace only).

        Returns:
            {
                "accepted":          bool,
                "score":             int,          # 1-10
                "rejection_reasons": List[str],
                "feedback":          str,          # critique; empty when accepted
                "_tool_calls":       List[Dict],
            }
        """
        t0 = time.perf_counter()
        h_id = verdict.get("hypothesis_id", hypothesis.get("id", "?"))

        messages = self._build_messages(verdict, hypothesis)
        raw = self._llm_call(messages)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        parsed = self._parse_response(raw)
        score = parsed["score"]
        flags = parsed["flags"]
        critique = parsed["critique"]

        accepted = score >= self.threshold
        rejection_reasons = _derive_rejection_reasons(flags) if not accepted else []

        return {
            "accepted":          accepted,
            "score":             score,
            "rejection_reasons": rejection_reasons,
            "feedback":          critique if not accepted else "",
            "_tool_calls":       [_reviewer_tool_call(h_id, attempt, score, accepted, flags, rejection_reasons, critique, latency_ms)],
        }

    # ── private: prompt building ──────────────────────────────────────────────

    def _build_messages(
        self,
        verdict: Dict[str, Any],
        hypothesis: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        evidence_text = _format_evidence(verdict.get("evidence", []))
        user_content = _USER_TEMPLATE.format(
            hypothesis=hypothesis.get("text", ""),
            label=verdict.get("label", ""),
            evidence=evidence_text,
            reasoning=verdict.get("reasoning", ""),
        )
        return [
            {"role": "system", "content": _REVIEWER_SYSTEM},
            {"role": "user",   "content": user_content},
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
            print(f"[ReviewerAgent] LLM call failed: {exc}")
            return ""

    # ── private: response parsing ─────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        fallback = {
            "score": 5,
            "flags": {
                "label_alignment":     "debatable",
                "evidence_quality":    "weak",
                "reasoning_coherence": "loose",
            },
            "critique": "Parse error — could not extract valid JSON from reviewer response.",
        }
        try:
            data = _extract_json(raw)
        except ValueError:
            return fallback

        score = int(data.get("score", 5))
        score = max(1, min(10, score))
        flags = data.get("flags", fallback["flags"])
        critique = str(data.get("critique", "")).strip()

        return {"score": score, "flags": flags, "critique": critique}


# ── module-level helpers ──────────────────────────────────────────────────────

def _derive_rejection_reasons(flags: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    la = flags.get("label_alignment", "")
    if la != "strong":
        reasons.append(f"label_alignment: {la}")
    eq = flags.get("evidence_quality", "")
    if eq not in ("strong", "not_required"):
        reasons.append(f"evidence_quality: {eq}")
    rc = flags.get("reasoning_coherence", "")
    if rc != "clear":
        reasons.append(f"reasoning_coherence: {rc}")
    return reasons


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    quotes = [
        f'  [{i}] "{span.get("quote", "").strip()}"'
        for i, span in enumerate(evidence, 1)
        if span.get("quote", "").strip()
    ]
    return "\n".join(quotes) if quotes else "(none)"


def _reviewer_tool_call(
    h_id: str,
    attempt: int,
    score: int,
    accepted: bool,
    flags: Dict[str, Any],
    rejection_reasons: List[str],
    critique: str,
    latency_ms: float,
) -> Dict[str, Any]:
    return {
        "name": "hypothesis_reviewer",
        "args": {"hypothesis_id": h_id, "attempt": attempt},
        "output": {
            "score":                 score,
            "accepted":              accepted,
            "label_alignment":       flags.get("label_alignment"),
            "evidence_quality":      flags.get("evidence_quality"),
            "reasoning_coherence":   flags.get("reasoning_coherence"),
            "rejection_reasons":     rejection_reasons,
            "critique":              critique,
        },
        "latency_ms": latency_ms,
    }


def _extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in: {text[:300]!r}")
