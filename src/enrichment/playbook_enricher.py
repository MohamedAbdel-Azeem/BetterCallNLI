"""
Playbook Enricher — Member 4, Milestone 3 (Task 4).

Applies the deterministic playbook policy (loaded verbatim from playbook.yaml)
to each raw verdict produced by HypothesisAnalyst. Produces an enriched verdict
shape consumed by RuntraceFormatter and the evaluation pipeline.

Design rules (per milestone3_plan.md + m3.pdf section 2d):

  * The playbook is integrated as-is — no edits to its file. We only READ it.
  * Severity + action resolution: start from `global_defaults.label_to_default_decision[label]`,
    then apply per-check `overrides[label]` if present.
  * Status mapping: `global_defaults.label_to_status[label]`
        ENTAILED      -> satisfied
        CONTRADICTED  -> conflict
        NOT_MENTIONED -> missing
  * Rationale is interpolated from `rationale_templates[status]` using the
    `template_slots` placeholders in playbook.yaml.
  * `playbook_rule_id` of the form  "<H_ID>_<LABEL>"  e.g.  "H04_NOT_MENTIONED"
  * Flags are computed deterministically from the verdict + resolved policy.

No LLM calls. Pure-Python deterministic policy layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── label -> playbook-status mapping (defensive fallback only) ───────────────
_DEFAULT_LABEL_TO_STATUS = {
    "ENTAILED":      "satisfied",
    "CONTRADICTED":  "conflict",
    "NOT_MENTIONED": "missing",
}

_DEFAULT_LABEL_TO_DECISION = {
    "ENTAILED":      {"severity": "LOW",    "action": "ACCEPT"},
    "CONTRADICTED":  {"severity": "HIGH",   "action": "ESCALATE"},
    "NOT_MENTIONED": {"severity": "MEDIUM", "action": "CLARIFY"},
}

_VALID_LABELS = set(_DEFAULT_LABEL_TO_STATUS.keys())


# ── safe string formatter — never raises on missing template keys ────────────
class _SafeDict(dict):
    def __missing__(self, key: str) -> str:          # pragma: no cover
        return "{" + key + "}"


# ── public API ───────────────────────────────────────────────────────────────

class PlaybookEnricher:
    """
    Stateless policy layer that turns raw Analyst verdicts into enriched
    verdicts ready for runtrace emission and end-user reporting.

    Args:
        playbook_path: path to playbook.yaml (default 'playbook.yaml' in repo root).

    Usage:
        enricher = PlaybookEnricher("playbook.yaml")
        enriched = enricher.enrich(pipeline_result["verdicts"])
    """

    def __init__(self, playbook_path: str = "playbook.yaml") -> None:
        path = Path(playbook_path)
        if not path.exists():
            raise FileNotFoundError(f"Playbook not found: {playbook_path}")

        with path.open("r", encoding="utf-8") as f:
            self.playbook: Dict[str, Any] = yaml.safe_load(f)

        self._checks_by_id: Dict[str, Dict[str, Any]] = {
            check["hypothesis_id"]: check
            for check in self.playbook.get("checks", [])
        }
        self._global_defaults: Dict[str, Any] = self.playbook.get("global_defaults", {})

    # ── public API ────────────────────────────────────────────────────────────

    def enrich(self, verdicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply playbook policy to each verdict. Returns a new list; inputs are not mutated."""
        return [self._enrich_one(v) for v in verdicts]

    def enrich_one(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Public single-verdict variant (mirrors enrich() for caller convenience)."""
        return self._enrich_one(verdict)

    # ── internals ─────────────────────────────────────────────────────────────

    def _enrich_one(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        h_id = verdict.get("hypothesis_id", "")
        label = str(verdict.get("label", "NOT_MENTIONED")).upper()
        if label not in _VALID_LABELS:
            label = "NOT_MENTIONED"

        check = self._checks_by_id.get(h_id)

        # Resolve severity + action: defaults -> per-check overrides
        defaults_block = self._global_defaults.get(
            "label_to_default_decision", _DEFAULT_LABEL_TO_DECISION
        )
        default_decision = defaults_block.get(label, _DEFAULT_LABEL_TO_DECISION[label])
        severity = default_decision.get("severity", "MEDIUM")
        action = default_decision.get("action", "CLARIFY")

        if check is not None:
            override = check.get("overrides", {}).get(label, {}) or {}
            severity = override.get("severity", severity)
            action = override.get("action", action)

        # Status mapping (satisfied / conflict / missing)
        status_map = self._global_defaults.get(
            "label_to_status", _DEFAULT_LABEL_TO_STATUS
        )
        status = status_map.get(label, _DEFAULT_LABEL_TO_STATUS[label])

        criticality = (check or {}).get("criticality", "P1")
        hypothesis_text = (check or {}).get("hypothesis_text", "")
        title = (check or {}).get("title", "")

        # Rationale interpolation
        rationale_templates = (check or {}).get("rationale_templates", {}) or {}
        template = rationale_templates.get(status, "")
        rationale = template.format_map(
            _SafeDict(
                HYPOTHESIS_TITLE=title,
                STATUS=status,
                SEVERITY=severity,
                ACTION=action,
                EVIDENCE_SUMMARY=self._summarise_evidence(verdict.get("evidence", [])),
                TOP_CITATION=self._top_citation(verdict.get("evidence", [])),
                CONFIDENCE=f"{float(verdict.get('confidence', 0.0)):.2f}",
            )
        )

        playbook_rule_id = f"{h_id}_{label}"

        flags = self._compute_flags(verdict, label, severity, status)

        # Preserve all original verdict keys; add hypothesis_text + risk block.
        return {
            **verdict,
            "hypothesis_text": hypothesis_text,
            "risk": {
                "severity":          severity,
                "action":            action,
                "status":            status,
                "criticality":       criticality,
                "playbook_rule_ids": [playbook_rule_id],
                "rationale":         rationale,
                "flags":             flags,
            },
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_flags(
        verdict: Dict[str, Any],
        label: str,
        severity: str,
        status: str,
    ) -> List[str]:
        """Deterministic policy flags derived from verdict + resolved policy."""
        flags: List[str] = []
        evidence = verdict.get("evidence", []) or []
        confidence = float(verdict.get("confidence", 0.0) or 0.0)

        # decisive label but no evidence cited
        if label in ("ENTAILED", "CONTRADICTED") and not evidence:
            flags.append("decisive_label_no_evidence")

        # high-severity + missing/decisive-no-evidence is worth surfacing
        if severity == "HIGH" and (status == "missing" or not evidence):
            flags.append("high_severity_unsupported")

        # decisive label with low confidence
        if label in ("ENTAILED", "CONTRADICTED") and confidence < 0.55:
            flags.append("low_confidence_decisive_label")

        # any evidence quote that could not be located verbatim in the contract
        if any(
            (span.get("char_start", -1) == -1)
            or (str(span.get("note", "")).startswith("not found"))
            for span in evidence
        ):
            flags.append("evidence_not_verbatim")

        return flags

    @staticmethod
    def _summarise_evidence(evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "(no contract evidence cited)"
        quotes = [str(span.get("quote", "")).strip() for span in evidence]
        quotes = [q for q in quotes if q]
        if not quotes:
            return "(no contract evidence cited)"
        first = quotes[0][:120]
        suffix = f" (+{len(quotes)-1} more)" if len(quotes) > 1 else ""
        return f'"{first}…"{suffix}' if len(quotes[0]) > 120 else f'"{first}"{suffix}'

    @staticmethod
    def _top_citation(evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return ""
        first = evidence[0]
        chunk_id = first.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        char_start = first.get("char_start", -1)
        if char_start >= 0:
            return f"char:{char_start}"
        return ""
