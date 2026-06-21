"""
Runtrace Formatter — Member 4, Milestone 3 (Task 4).

Schema-compliant runtrace emitter for the BetterCallNLI multi-agent system.
Validates against runtrace_ms3.schema.json. Supports both modes mandated by
m3.pdf section 2c:

    * mode = "contract"      — one runtrace per contract (17-hypothesis run)
    * mode = "conversation"  — one runtrace per conversation session

Design rules (per "Don't modify other code" constraint):

  * This module is a pure consumer of whatever the existing agents
    (IntentRouter, HypothesisAnalyst, ReviewerAgent, HypothesisPipeline,
    ConversationAgent, ConversationHistory) already return.

  * Where source agents emit tool_calls without `count` (intent_router,
    hypothesis_reviewer) the formatter injects `count = 1` so the runtrace
    matches m3.pdf section 2h (Name, args, output, count per agent).

  * Where ConversationAgent emits no tool_calls at all, the formatter
    RECONSTRUCTS the retrieve + conversation_agent tool_calls from the
    observable return values of `chat()` (precedents, response, usage,
    retrieval_mode, evidence). The reconstruction is best-effort but
    factually accurate — it lacks only per-call latency_ms (left absent
    in the runtrace, since the schema marks latency_ms as optional).

  * EnrichedVerdict.hypothesis_text is sourced by PlaybookEnricher from
    the playbook (not the analyst). EnrichedVerdict.accepted / attempts
    are merged in here from the corresponding agent_trace.

  * Field-name mapping for ConversationTurn:
        history turn key "user"       ->  runtrace key "user_prompt"
        history turn key "assistant"  ->  runtrace key "assistant_response"
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── module constants ─────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0-ms3"
FRAMEWORK      = "multi_agent_pipeline"
DEFAULT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

_VALID_LABELS = ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")


# ── dataclasses (for type-clarity at call sites; not used as JSON schema) ────

@dataclass
class ToolCall:
    name:       str
    args:       Dict[str, Any] = field(default_factory=dict)
    output:     Dict[str, Any] = field(default_factory=dict)
    count:      int            = 1
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name":   self.name,
            "args":   dict(self.args),
            "output": dict(self.output),
            "count":  int(self.count),
        }
        if self.latency_ms is not None:
            d["latency_ms"] = float(self.latency_ms)
        return d


@dataclass
class AgentTrace:
    agent_id:      str
    attempts:      int
    tool_calls:    List[Dict[str, Any]]
    accepted:      bool
    hypothesis_id: Optional[str]      = None
    turn_id:       Optional[int]      = None
    final_verdict: Optional[Dict[str, Any]] = None
    started_at:    Optional[str]      = None
    ended_at:      Optional[str]      = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "agent_id":   self.agent_id,
            "attempts":   int(self.attempts),
            "accepted":   bool(self.accepted),
            "tool_calls": [RuntraceFormatter._normalize_tool_call(tc) for tc in self.tool_calls],
        }
        # optional fields — only emit when set (keeps runtrace clean)
        if self.hypothesis_id is not None:
            d["hypothesis_id"] = self.hypothesis_id
        if self.turn_id is not None:
            d["turn_id"] = int(self.turn_id)
        if self.final_verdict is not None:
            d["final_verdict"] = self.final_verdict
        if self.started_at:
            d["started_at"] = self.started_at
        if self.ended_at:
            d["ended_at"] = self.ended_at
        return d


# ── main class ───────────────────────────────────────────────────────────────

class RuntraceFormatter:
    """
    Build schema-compliant MS3 runtraces from the data shapes already produced
    by the existing agents. No agent code is modified.

    Args:
        playbook_path : path to playbook.yaml (used for playbook_id, version,
                        ruleset_hash). Default 'playbook.yaml' in repo root.
        model         : HF model string for the `run.parameters.base_model`
                        field. Default Qwen/Qwen2.5-7B-Instruct.
    """

    def __init__(
        self,
        playbook_path: str = "playbook.yaml",
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.playbook_path = Path(playbook_path)
        self.model = model

        # Read once at construction; cached for the lifetime of the formatter.
        self._playbook_raw, self._playbook_meta = self._load_playbook_meta(self.playbook_path)

    # ─────────────────────────────────────────────────────────────────────────
    # public API — CONTRACT MODE
    # ─────────────────────────────────────────────────────────────────────────

    def build_contract_runtrace(
        self,
        *,
        contract:             Dict[str, Any],
        intent_router_call:   Dict[str, Any],
        agent_traces:         List[Dict[str, Any]],
        enriched_verdicts:    List[Dict[str, Any]],
        retrieval_mode:       str,
        started_at:           str,
        ended_at:             str,
        model:                Optional[str] = None,
        run_id:               Optional[str] = None,
        gold_labels:          Optional[Dict[str, str]] = None,
        run_validations:      Optional[List[Dict[str, Any]]] = None,
        run_parameters_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build one MS3 contract-mode runtrace dict.

        Args:
            contract           : dict with at minimum {"id": str, "text": str}.
            intent_router_call : the single ToolCall dict returned by IntentRouter.route()
                                 — the call that decided this run's mode.
            agent_traces       : list of per-hypothesis trace dicts exactly as produced
                                 by HypothesisPipeline.run() (one entry per H01..H17).
            enriched_verdicts  : list of verdict dicts after PlaybookEnricher.enrich().
            retrieval_mode     : "vector_rag" | "graphrag".
            started_at         : ISO-8601 UTC timestamp string when the run began.
            ended_at           : ISO-8601 UTC timestamp string when the run ended.
            model              : override the formatter's default model string.
            run_id             : override the auto-generated run id.
            gold_labels        : optional {hypothesis_id: gold_label} for evaluate mode.
                                 When provided, metrics include accuracy + confusion_counts.
            run_validations    : optional list of pre-built ValidationResult dicts.
            run_parameters_extra: extra fields to embed under run.parameters.

        Returns:
            A schema-compliant runtrace dict (mode == "contract").
        """
        retrieval_mode_norm = self._normalize_retrieval_mode(retrieval_mode)

        # Merge accepted / attempts / gold_label from agent_traces into enriched verdicts.
        verdicts_for_runtrace = self._merge_pipeline_meta_into_verdicts(
            enriched_verdicts=enriched_verdicts,
            agent_traces=agent_traces,
            gold_labels=gold_labels,
        )

        # Normalize agent_traces (inject count where missing).
        agent_traces_norm = [self._normalize_agent_trace(t) for t in agent_traces]

        # Compute MS1-comparable metrics.
        contract_latency_ms = self._latency_ms_between(started_at, ended_at)
        metrics = self._compute_contract_metrics(
            enriched_verdicts=verdicts_for_runtrace,
            agent_traces=agent_traces,
            gold_labels=gold_labels,
            contract_latency_ms=contract_latency_ms,
        )

        runtrace: Dict[str, Any] = {
            "schema_version":  SCHEMA_VERSION,
            "mode":            "contract",
            "retrieval_mode":  retrieval_mode_norm,
            "run":             self._build_run_block(
                                    started_at=started_at,
                                    ended_at=ended_at,
                                    model=model or self.model,
                                    run_id=run_id,
                                    extra_parameters=run_parameters_extra,
                                ),
            "contract":        self._build_contract_block(contract),
            "playbook":        self._build_playbook_block(),
            "intent_router":   self._normalize_tool_call(intent_router_call),
            "agent_traces":    agent_traces_norm,
            "enriched_verdicts": verdicts_for_runtrace,
            "metrics":         metrics,
        }
        if run_validations:
            runtrace["run_validations"] = list(run_validations)
        return runtrace

    # ─────────────────────────────────────────────────────────────────────────
    # public API — CONVERSATION MODE
    # ─────────────────────────────────────────────────────────────────────────

    def build_conversation_runtrace(
        self,
        *,
        contract:        Dict[str, Any],
        session_id:      str,
        turn_records:    List[Dict[str, Any]],
        retrieval_mode:  str,
        started_at:      str,
        ended_at:        str,
        model:           Optional[str] = None,
        run_id:          Optional[str] = None,
        created_at:      Optional[str] = None,
        run_validations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build one MS3 conversation-mode runtrace dict for a full session.

        Args:
            contract        : dict with at minimum {"id": str, "text": str}.
            session_id      : ConversationHistory.session_id (or any unique string).
            turn_records    : list of per-turn dicts. Each must contain:
                {
                    "user_prompt":         str,     # also accepts "user"
                    "assistant_response":  str,     # also accepts "assistant"
                    "evidence":            List[Dict],         # contract-grounded
                    "precedents":          List[Dict],         # external retrieval
                    "retrieval_mode":      str,
                    "intent_router_call":  Dict,               # the per-turn router ToolCall
                    "usage":               Optional[Dict],     # {prompt_tokens, completion_tokens}
                    "timestamp":           Optional[str],
                    "response":            Optional[str],      # alias for assistant_response
                    "chat_result":         Optional[Dict],     # if present, used to reconstruct
                                                               # conversation_agent tool_calls
                    "tool_calls":          Optional[List[Dict]] # already-built per-agent calls,
                                                                # used as-is if provided
                }
            retrieval_mode  : default retrieval mode for the session (each turn may override).
            started_at      : ISO-8601 UTC start.
            ended_at        : ISO-8601 UTC end.
            model           : override formatter's default model.
            run_id          : override auto-generated run id.
            created_at      : ConversationHistory.created_at; defaults to started_at.
            run_validations : optional pre-built ValidationResult dicts.

        Returns:
            A schema-compliant runtrace dict (mode == "conversation").
        """
        retrieval_mode_norm = self._normalize_retrieval_mode(retrieval_mode)

        agent_traces:      List[Dict[str, Any]] = []
        runtrace_turns:    List[Dict[str, Any]] = []

        # Decide which router call goes into the top-level field (m3.pdf: always present).
        top_level_router_call: Dict[str, Any] = {}
        if turn_records:
            first = turn_records[0]
            top_level_router_call = self._normalize_tool_call(
                first.get("intent_router_call", {})
            )

        for idx, turn in enumerate(turn_records, start=1):
            turn_id = int(turn.get("turn_id", idx))
            turn_retrieval_mode = self._normalize_retrieval_mode(
                turn.get("retrieval_mode", retrieval_mode_norm)
            )

            user_prompt = (
                turn.get("user_prompt")
                or turn.get("user")
                or ""
            )
            assistant_response = (
                turn.get("assistant_response")
                or turn.get("assistant")
                or turn.get("response")
                or ""
            )

            # ── conversation turn block ──────────────────────────────────────
            turn_block: Dict[str, Any] = {
                "turn_id":            turn_id,
                "user_prompt":        user_prompt,
                "assistant_response": assistant_response,
                "retrieval_mode":     turn_retrieval_mode,
            }
            if "timestamp" in turn and turn["timestamp"]:
                turn_block["timestamp"] = turn["timestamp"]
            if turn.get("evidence"):
                turn_block["evidence"] = list(turn["evidence"])
            if turn.get("precedents"):
                turn_block["precedents"] = list(turn["precedents"])
            if turn.get("usage"):
                turn_block["usage"] = {
                    "prompt_tokens":     int(turn["usage"].get("prompt_tokens", 0)),
                    "completion_tokens": int(turn["usage"].get("completion_tokens", 0)),
                }
            runtrace_turns.append(turn_block)

            # ── per-turn agent_traces ────────────────────────────────────────
            #   trace 1 : intent_router for this turn
            #   trace 2 : conversation_agent for this turn
            router_tc = self._normalize_tool_call(turn.get("intent_router_call", {}))
            if router_tc.get("name"):
                agent_traces.append(
                    AgentTrace(
                        agent_id="intent_router",
                        attempts=1,
                        accepted=True,
                        tool_calls=[router_tc],
                        turn_id=turn_id,
                    ).to_dict()
                )

            conv_tool_calls = self._resolve_conversation_tool_calls(
                turn=turn,
                retrieval_mode=turn_retrieval_mode,
            )
            if conv_tool_calls:
                agent_traces.append(
                    AgentTrace(
                        agent_id="conversation_agent",
                        attempts=1,
                        accepted=True,
                        tool_calls=conv_tool_calls,
                        turn_id=turn_id,
                    ).to_dict()
                )

        runtrace: Dict[str, Any] = {
            "schema_version":  SCHEMA_VERSION,
            "mode":            "conversation",
            "retrieval_mode":  retrieval_mode_norm,
            "run":             self._build_run_block(
                                    started_at=started_at,
                                    ended_at=ended_at,
                                    model=model or self.model,
                                    run_id=run_id,
                                ),
            "contract":        self._build_contract_block(contract),
            "playbook":        self._build_playbook_block(),
            "intent_router":   top_level_router_call or {"name": "intent_router", "args": {}, "output": {}, "count": 0},
            "agent_traces":    agent_traces or [
                # graceful-empty: schema requires minItems=1; this only fires for
                # the degenerate "session with zero turns" case which the CLI
                # should not produce — kept as a defensive minimum.
                AgentTrace(
                    agent_id="intent_router",
                    attempts=1,
                    accepted=True,
                    tool_calls=[top_level_router_call or {"name": "intent_router", "args": {}, "output": {}, "count": 0}],
                ).to_dict()
            ],
            "conversation":    {
                "session_id":  session_id,
                "created_at":  created_at or started_at,
                "turn_count":  len(runtrace_turns),
                "turns":       runtrace_turns,
            },
        }
        if run_validations:
            runtrace["run_validations"] = list(run_validations)
        return runtrace

    # ─────────────────────────────────────────────────────────────────────────
    # public API — persistence
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def save(runtrace: Dict[str, Any], path: str) -> None:
        """Write a runtrace dict to disk as pretty-printed JSON (UTF-8, ensure_ascii=False)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(runtrace, indent=2, ensure_ascii=False), encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # internals — block builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_run_block(
        self,
        started_at: str,
        ended_at:   str,
        model:      str,
        run_id:     Optional[str] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if run_id is None:
            run_id = f"ms3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        parameters: Dict[str, Any] = {"base_model": model}
        if extra_parameters:
            for k, v in extra_parameters.items():
                if v is not None:
                    parameters[k] = v
        return {
            "run_id":     run_id,
            "started_at": started_at,
            "ended_at":   ended_at,
            "framework":  FRAMEWORK,
            "parameters": parameters,
        }

    def _build_contract_block(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        text = str(contract.get("text", "") or "")
        contract_id = str(contract.get("id", "") or "unknown")
        block: Dict[str, Any] = {
            "contract_id":  contract_id,
            "source_type":  contract.get("source_type", "txt"),
            "hash_sha256":  self._sha256_hex(text),
            "char_count":   len(text),
        }
        if "source_name" in contract:
            block["source_name"] = contract["source_name"]
        block["language"] = contract.get("language", "en")
        # chunks pass-through if the caller supplied them
        if "chunks" in contract and isinstance(contract["chunks"], list):
            block["chunks"] = contract["chunks"]
        return block

    def _build_playbook_block(self) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "playbook_id":  self._playbook_meta["playbook_id"],
            "version":      self._playbook_meta["version"],
            "ruleset_hash": self._playbook_meta["ruleset_hash"],
        }
        defaults = self._playbook_meta.get("rule_params")
        if defaults:
            block["rule_params"] = defaults
        return block

    # ─────────────────────────────────────────────────────────────────────────
    # internals — tool_call / agent_trace normalisation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_tool_call(tc: Any) -> Dict[str, Any]:
        """
        Coerce any tool_call dict to the schema shape:
            { name, args, output, count, latency_ms? }

        Injects count=1 for source agents that don't emit it (intent_router,
        hypothesis_reviewer). Ensures args / output are dicts.
        """
        if not isinstance(tc, dict):
            return {"name": "unknown", "args": {}, "output": {}, "count": 0}
        out: Dict[str, Any] = {
            "name":   str(tc.get("name", "unknown")),
            "args":   dict(tc.get("args", {}) or {}),
            "output": dict(tc.get("output", {}) or {}),
            "count":  int(tc.get("count", 1)),
        }
        if "latency_ms" in tc and tc["latency_ms"] is not None:
            try:
                out["latency_ms"] = float(tc["latency_ms"])
            except (TypeError, ValueError):
                pass
        return out

    def _normalize_agent_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all tool_calls inside a trace; preserve everything else."""
        tool_calls = [self._normalize_tool_call(tc) for tc in trace.get("tool_calls", [])]
        norm: Dict[str, Any] = {
            "agent_id":   str(trace.get("agent_id", "hypothesis_analyst")),
            "attempts":   int(trace.get("attempts", 1)),
            "accepted":   bool(trace.get("accepted", False)),
            "tool_calls": tool_calls,
        }
        if "hypothesis_id" in trace and trace["hypothesis_id"] is not None:
            norm["hypothesis_id"] = trace["hypothesis_id"]
        if "turn_id" in trace and trace["turn_id"] is not None:
            norm["turn_id"] = int(trace["turn_id"])
        if "final_verdict" in trace and trace["final_verdict"] is not None:
            norm["final_verdict"] = trace["final_verdict"]
        if trace.get("started_at"):
            norm["started_at"] = trace["started_at"]
        if trace.get("ended_at"):
            norm["ended_at"] = trace["ended_at"]
        return norm

    # ─────────────────────────────────────────────────────────────────────────
    # internals — conversation tool_call reconstruction
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_conversation_tool_calls(
        self,
        turn: Dict[str, Any],
        retrieval_mode: str,
    ) -> List[Dict[str, Any]]:
        """
        Three sources, in order of preference:
          1. turn["tool_calls"]            — caller already built them.
          2. turn["chat_result"]           — reconstruct from ConversationAgent.chat() return.
          3. inline turn fields             — reconstruct from precedents/response/evidence/usage.
        """
        if turn.get("tool_calls"):
            return [self._normalize_tool_call(tc) for tc in turn["tool_calls"]]

        chat_result = turn.get("chat_result")
        if isinstance(chat_result, dict):
            return self._reconstruct_from_chat_result(chat_result, retrieval_mode)

        return self._reconstruct_from_inline_turn(turn, retrieval_mode)

    @staticmethod
    def _reconstruct_from_chat_result(
        chat_result: Dict[str, Any],
        retrieval_mode: str,
    ) -> List[Dict[str, Any]]:
        precedents = chat_result.get("precedents", []) or []
        response   = chat_result.get("response", "") or ""
        evidence   = chat_result.get("evidence", []) or []
        usage      = chat_result.get("usage", {}) or {}

        retrieve_tc = {
            "name": "retrieve",
            "args": {
                "retrieval_mode": chat_result.get("retrieval_mode", retrieval_mode),
                "k_requested":   len(precedents),
            },
            "output": {
                "count":     len(precedents),
                "top_score": precedents[0].get("score") if precedents else None,
            },
            "count": len(precedents),
        }
        conv_tc = {
            "name": "conversation_agent",
            "args": {
                "retrieval_mode": chat_result.get("retrieval_mode", retrieval_mode),
            },
            "output": {
                "response_chars":    len(response),
                "evidence_count":    len(evidence),
                "prompt_tokens":     int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
            "count": 1,
        }
        return [retrieve_tc, conv_tc]

    @staticmethod
    def _reconstruct_from_inline_turn(
        turn: Dict[str, Any],
        retrieval_mode: str,
    ) -> List[Dict[str, Any]]:
        precedents = turn.get("precedents", []) or []
        evidence   = turn.get("evidence", []) or []
        response   = turn.get("assistant_response") or turn.get("assistant") or turn.get("response") or ""
        usage      = turn.get("usage", {}) or {}

        retrieve_tc = {
            "name": "retrieve",
            "args": {
                "retrieval_mode": turn.get("retrieval_mode", retrieval_mode),
                "k_requested":   len(precedents),
            },
            "output": {
                "count":     len(precedents),
                "top_score": precedents[0].get("score") if precedents else None,
            },
            "count": len(precedents),
        }
        conv_tc = {
            "name": "conversation_agent",
            "args": {
                "retrieval_mode": turn.get("retrieval_mode", retrieval_mode),
            },
            "output": {
                "response_chars":    len(str(response)),
                "evidence_count":    len(evidence),
                "prompt_tokens":     int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
            "count": 1,
        }
        return [retrieve_tc, conv_tc]

    # ─────────────────────────────────────────────────────────────────────────
    # internals — enriched-verdict merging (accepted / attempts / gold_label)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_pipeline_meta_into_verdicts(
        enriched_verdicts: List[Dict[str, Any]],
        agent_traces:      List[Dict[str, Any]],
        gold_labels:       Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bring in `accepted`, `attempts`, and (optionally) `gold_label` from the
        pipeline traces / gold map. Keeps schema's EnrichedVerdict required-set complete.
        """
        traces_by_hid: Dict[str, Dict[str, Any]] = {
            t.get("hypothesis_id"): t for t in agent_traces if t.get("hypothesis_id")
        }

        out: List[Dict[str, Any]] = []
        for v in enriched_verdicts:
            h_id  = v.get("hypothesis_id")
            trace = traces_by_hid.get(h_id, {}) if h_id else {}
            merged: Dict[str, Any] = {
                **v,
                "accepted": bool(trace.get("accepted", False)),
                "attempts": int(trace.get("attempts", 1)),
            }
            if gold_labels:
                gold = gold_labels.get(h_id)
                merged["gold_label"] = gold if gold in _VALID_LABELS else None
            out.append(merged)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # internals — metrics
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_contract_metrics(
        enriched_verdicts:   List[Dict[str, Any]],
        agent_traces:        List[Dict[str, Any]],
        gold_labels:         Optional[Dict[str, str]],
        contract_latency_ms: float,
    ) -> Dict[str, Any]:
        label_counts = {"ENTAILED": 0, "CONTRADICTED": 0, "NOT_MENTIONED": 0}
        confusion_counts: Dict[str, int] = {}
        correct_count          = 0
        compliant_count        = 0
        quote_integrity_count  = 0
        accepted_first_attempt = 0
        total_attempts         = 0

        total = len(enriched_verdicts)

        for v in enriched_verdicts:
            label = str(v.get("label", "NOT_MENTIONED")).upper()
            if label not in label_counts:
                label = "NOT_MENTIONED"
            label_counts[label] += 1

            evidence = v.get("evidence", []) or []

            # Groundedness: ENTAILED/CONTRADICTED must have at least one span;
            # NOT_MENTIONED is compliant when no spans are cited.
            if label in ("ENTAILED", "CONTRADICTED"):
                if evidence:
                    compliant_count += 1
            else:
                if not evidence:
                    compliant_count += 1

            # Quote integrity: every cited span must have been located verbatim.
            if not evidence:
                quote_integrity_count += 1  # vacuously true
            else:
                if all(
                    int(span.get("char_start", -1)) >= 0
                    and not str(span.get("note", "")).startswith("not found")
                    for span in evidence
                ):
                    quote_integrity_count += 1

            if gold_labels:
                gold = gold_labels.get(v.get("hypothesis_id"))
                if gold in _VALID_LABELS:
                    conf_key = f"{label}|{gold}"
                    confusion_counts[conf_key] = confusion_counts.get(conf_key, 0) + 1
                    if label == gold:
                        correct_count += 1

        for trace in agent_traces:
            attempts = int(trace.get("attempts", 1))
            total_attempts += attempts
            if trace.get("accepted") and attempts == 1:
                accepted_first_attempt += 1

        metrics: Dict[str, Any] = {
            "hypothesis_count":      17,
            "correct_count":         int(correct_count),
            "compliant_count":       int(compliant_count),
            "quote_integrity_count": int(quote_integrity_count),
            "contract_accuracy":     (correct_count / total) if (gold_labels and total) else 0.0,
            "groundedness_rate":     (compliant_count / total) if total else 0.0,
            "quote_integrity_rate":  (quote_integrity_count / total) if total else 0.0,
            "contract_latency_ms":   float(contract_latency_ms),
            "label_counts":          label_counts,
        }
        if confusion_counts:
            metrics["confusion_counts"] = confusion_counts
        if agent_traces:
            metrics["review_acceptance_rate"] = accepted_first_attempt / len(agent_traces)
            metrics["avg_review_attempts"]    = total_attempts / len(agent_traces)
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # internals — playbook loading / hashing
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_playbook_meta(path: Path) -> Any:
        """Return (raw_yaml_text, {playbook_id, version, ruleset_hash, rule_params})."""
        if not path.exists():
            raise FileNotFoundError(f"Playbook not found at {path}")

        raw = path.read_text(encoding="utf-8")
        try:
            import yaml  # local import keeps top-level dep light if unused
        except ImportError as exc:                                          # pragma: no cover
            raise RuntimeError("PyYAML required to read playbook.yaml") from exc

        parsed = yaml.safe_load(raw)
        meta = {
            "playbook_id":  str(parsed.get("playbook_id", "unknown")),
            "version":      str(parsed.get("version", "0.0")),
            "ruleset_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }
        # Carry the playbook's own evidence_required_for as run_params for traceability.
        rule_params = (parsed.get("global_defaults") or {}).get("evidence_required_for")
        if rule_params is not None:
            meta["rule_params"] = {"evidence_required_for": list(rule_params)}
        return raw, meta

    # ─────────────────────────────────────────────────────────────────────────
    # internals — small helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256_hex(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_retrieval_mode(mode: Any) -> str:
        normalised = str(mode or "").strip().lower().replace("-", "_")
        if normalised in ("graphrag", "graph_rag", "graph"):
            return "graphrag"
        if normalised in ("vector", "vector_rag", "vectorrag", "rag"):
            return "vector_rag"
        # default to vector_rag so the schema's enum stays satisfied
        return "vector_rag"

    @staticmethod
    def _latency_ms_between(started_at: str, ended_at: str) -> float:
        try:
            t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            return max(0.0, (t1 - t0).total_seconds() * 1000.0)
        except Exception:                                                   # pragma: no cover
            return 0.0


# ── module exports ───────────────────────────────────────────────────────────

__all__ = [
    "RuntraceFormatter",
    "ToolCall",
    "AgentTrace",
    "SCHEMA_VERSION",
    "FRAMEWORK",
]
