"""
MS3 evaluation runner — runs the full Router → Analyst → Reviewer pipeline on
the ContractNLI test split (MS3 spec §3) and computes the same metrics MS1 used:

    - label_accuracy            (per hypothesis + overall)
    - groundedness_rate         (% of evidence spans that locate verbatim in the contract)
    - quote_integrity_rate      (% of spans where contract[char_start:char_end] == quote)
    - avg_latency_ms            (per contract)
    - confusion_counts          (pred|gold pairs)
    - label_counts              (per predicted label)

Each verdict is also enriched with deterministic playbook outputs (§3c):
    status, severity, action, criticality, rationale

Outputs (per --output-dir):

    predictions_ms3.json                one record per (contract, hypothesis)
    runtraces/runtrace_<id>.json        one runtrace per contract (§2c, §2h)
    evaluation_metrics_ms3.csv          MS3 row, header matches MS1 CSV
    evaluation_metrics_combined.csv     MS1 + MS3 side-by-side (§5b)
    evaluation_metrics_ms3.json         full breakdown (per-hypothesis + confusion)
    runtraces_ms3.zip                   zip of every runtrace (§5c deliverable)

The fine-tuned MS1 model is NOT used (§2f); the agents call HuggingFace Serverless
with a same-family base model (Qwen/Qwen2.5-7B-Instruct by default).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Make the package (src/) and repo root importable so the soft imports below and
# the sibling-script imports resolve whether this file is run as
# `python scripts/evaluate_ms3.py`, `python -m scripts.evaluate_ms3`, or imported
# by apps/cli.py. (Harmless no-op inside the flattened Kaggle bundle.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Task 4 modules: prefer them when available, fall back to local shims ─────
# These are owned by Member 4 (playbook enrichment + schema-compliant runtrace
# formatter). When their PR merges into this branch the imports succeed and the
# runner automatically routes through their implementations. Until then, the
# local apply_playbook() / write_runtrace() defined below are used.

try:
    from bettercallnli.enrichment.playbook_enricher import PlaybookEnricher  # type: ignore
    _HAS_TASK4_ENRICHER = True
except Exception:
    PlaybookEnricher = None  # type: ignore[assignment]
    _HAS_TASK4_ENRICHER = False

try:
    from bettercallnli.utils.runtrace import RuntraceFormatter  # type: ignore
    _HAS_TASK4_FORMATTER = True
except Exception:
    RuntraceFormatter = None  # type: ignore[assignment]
    _HAS_TASK4_FORMATTER = False


# ── ContractNLI hypothesis-id mapping (matches src/retrieval/vector_rag.py) ────
H_TO_NDA: Dict[str, str] = {
    "H01": "nda-1",  "H02": "nda-2",  "H03": "nda-3",
    "H04": "nda-4",  "H05": "nda-5",  "H06": "nda-7",
    "H07": "nda-8",  "H08": "nda-10", "H09": "nda-11",
    "H10": "nda-12", "H11": "nda-13", "H12": "nda-15",
    "H13": "nda-16", "H14": "nda-17", "H15": "nda-18",
    "H16": "nda-19", "H17": "nda-20",
}

VALID_LABELS = ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")


# ── gold label extraction ─────────────────────────────────────────────────────

def extract_gold_labels(contract: Dict[str, Any]) -> Dict[str, str]:
    """
    Pull the ContractNLI annotation choice for every hypothesis the dataset
    covers. Returns {H_id: gold_label} for the 17 hypotheses in our playbook.

    The raw dataset uses keys like "nda-1" inside annotation_sets[0]["annotations"];
    we translate back to H01–H17 to align with the playbook + analyst outputs.
    """
    out: Dict[str, str] = {}
    ann_sets = contract.get("annotation_sets") or []
    if not ann_sets:
        return out
    annotations = ann_sets[0].get("annotations", {}) or {}

    for h_id, nda_id in H_TO_NDA.items():
        entry = annotations.get(nda_id)
        if not entry:
            continue
        choice = entry.get("choice", "")
        gold = _normalise_label(choice)
        if gold in VALID_LABELS:
            out[h_id] = gold

    return out


def _normalise_label(raw: str) -> str:
    """
    ContractNLI uses {"Entailment", "Contradiction", "NotMentioned"}, while
    MS1 and our pipeline emit {"ENTAILED", "CONTRADICTED", "NOT_MENTIONED"}.
    Normalise either form to the MS1 convention.
    """
    s = (raw or "").strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    if s in ("ENTAILMENT", "ENTAILED"):
        return "ENTAILED"
    if s in ("CONTRADICTION", "CONTRADICTED"):
        return "CONTRADICTED"
    if s in ("NOTMENTIONED", "NOMENTION"):
        return "NOT_MENTIONED"
    return s


# ── per-verdict scoring ───────────────────────────────────────────────────────

def score_verdict(
    verdict: Dict[str, Any],
    gold_label: str,
    contract_text: str,
) -> Dict[str, Any]:
    """
    Return per-verdict diagnostics:
        pred_label, gold_label, correct,
        groundedness_pass, quote_integrity_pass,
        evidence_count
    """
    pred_label = (verdict.get("label") or "").strip().upper()
    correct = pred_label == gold_label

    spans = verdict.get("evidence", []) or []

    # groundedness — every span has a non-(-1, -1) position
    grounded_spans = [s for s in spans if s.get("char_start", -1) >= 0]
    groundedness_pass = (
        len(grounded_spans) == len(spans) and len(spans) > 0
        if pred_label in ("ENTAILED", "CONTRADICTED")
        else True  # NOT_MENTIONED with no evidence is vacuously grounded
    )

    # quote integrity — contract[start:end] == quote, char-for-char
    integrity_results: List[bool] = []
    for s in spans:
        cs = int(s.get("char_start", -1))
        ce = int(s.get("char_end", -1))
        quote = s.get("quote", "")
        if cs < 0 or ce < 0 or cs >= len(contract_text) or ce > len(contract_text):
            integrity_results.append(False)
            continue
        integrity_results.append(contract_text[cs:ce] == quote)
    quote_integrity_pass = all(integrity_results) and bool(integrity_results) \
        if pred_label in ("ENTAILED", "CONTRADICTED") else True

    return {
        "pred_label":           pred_label,
        "gold_label":           gold_label,
        "correct":              correct,
        "groundedness_pass":    groundedness_pass,
        "quote_integrity_pass": quote_integrity_pass,
        "evidence_count":       len(spans),
    }


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_metrics(
    per_verdict: List[Dict[str, Any]],
    latencies: List[float],
) -> Dict[str, Any]:
    n = len(per_verdict)
    if n == 0:
        return {
            "contract_count":       0,
            "hypothesis_count":     0,
            "label_accuracy":       0.0,
            "groundedness_rate":    0.0,
            "quote_integrity_rate": 0.0,
            "avg_latency_ms":       0.0,
            "label_counts":         {},
            "confusion_counts":     {},
            "per_hypothesis":       {},
        }

    correct = sum(1 for v in per_verdict if v["correct"])
    grounded = sum(1 for v in per_verdict if v["groundedness_pass"])
    integrity = sum(1 for v in per_verdict if v["quote_integrity_pass"])

    label_counts: Dict[str, int] = {}
    confusion_counts: Dict[str, int] = {}
    per_hyp: Dict[str, Dict[str, Any]] = {}

    for v in per_verdict:
        label_counts[v["pred_label"]] = label_counts.get(v["pred_label"], 0) + 1
        key = f"{v['pred_label']}|{v['gold_label']}"
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

        h_id = v.get("hypothesis_id", "?")
        bucket = per_hyp.setdefault(h_id, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if v["correct"]:
            bucket["correct"] += 1

    for h_id, bucket in per_hyp.items():
        bucket["accuracy"] = (
            round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0
        )

    return {
        "contract_count":       len({v["contract_id"] for v in per_verdict}),
        "hypothesis_count":     n,
        "label_accuracy":       round(correct / n, 4),
        "groundedness_rate":    round(grounded / n, 4),
        "quote_integrity_rate": round(integrity / n, 4),
        "avg_latency_ms":       round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "label_counts":         label_counts,
        "confusion_counts":     confusion_counts,
        "per_hypothesis":       per_hyp,
    }


# ── playbook (deterministic mapping, MS3 §3c) ────────────────────────────────

def load_playbook(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_playbook(verdict: Dict[str, Any], playbook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically derive policy fields from `playbook.yaml` and merge them into
    the verdict. Pure code — no LLM. Required by MS3 §3c.

    Adds:
        status         : satisfied | conflict | missing   (from label_to_status)
        severity       : LOW | MEDIUM | HIGH              (default + per-hypothesis override)
        action         : ACCEPT | NEGOTIATE | CLARIFY | ESCALATE
        criticality    : P0 | P1 | P2
        rationale      : rendered from rationale_templates[status]
    """
    h_id = verdict.get("hypothesis_id", "")
    label = (verdict.get("label") or "").upper()

    defaults = playbook.get("global_defaults", {}) or {}
    label_to_status   = defaults.get("label_to_status", {}) or {}
    default_decisions = defaults.get("label_to_default_decision", {}) or {}

    rule = next(
        (c for c in playbook.get("checks", []) if c.get("hypothesis_id") == h_id),
        {},
    )

    # decision = default for label, then overridden by per-check entry
    decision = dict(default_decisions.get(label, {}))
    decision.update((rule.get("overrides") or {}).get(label, {}))

    status = label_to_status.get(label, "missing")
    title  = rule.get("title", h_id)
    severity = decision.get("severity", "MEDIUM")
    action   = decision.get("action",   "CLARIFY")

    rationale_tpl = (rule.get("rationale_templates") or {}).get(status, "")
    rationale = (
        rationale_tpl
        .replace("{HYPOTHESIS_TITLE}", title)
        .replace("{STATUS}",           status)
        .replace("{SEVERITY}",         severity)
        .replace("{ACTION}",           action)
        .replace("{EVIDENCE_SUMMARY}", _short_evidence(verdict))
        .replace("{CONFIDENCE}",       f"{float(verdict.get('confidence', 0.0)):.2f}")
    )

    return {
        **verdict,
        "status":        status,
        "severity":      severity,
        "action":        action,
        "criticality":   rule.get("criticality", "P2"),
        "rationale":     rationale,
    }


def _short_evidence(verdict: Dict[str, Any]) -> str:
    spans = verdict.get("evidence", []) or []
    if not spans:
        return "no relevant clause located"
    first = spans[0].get("quote", "").strip()
    return (first[:120] + "…") if len(first) > 120 else first


# ── tool-call normalization (MS3 §2h) ────────────────────────────────────────

def normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    §2h requires every tool_call to carry {name, args, output, count}.
    Agents emit name/args/output and most include a count. We backfill `count`
    where the agent omits it (default 1 — one invocation per tool_call entry).
    """
    out = []
    for tc in tool_calls:
        out.append({
            "name":   tc.get("name", "unknown"),
            "args":   tc.get("args", {}),
            "output": tc.get("output", {}),
            "count":  int(tc.get("count", 1)),
            **({"latency_ms": tc["latency_ms"]} if "latency_ms" in tc else {}),
        })
    return out


# ── output writers ────────────────────────────────────────────────────────────

def write_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Match the MS1 CSV header so the two runs can be diffed directly."""
    header = "label_accuracy,groundedness,quote_integrity_pass_rate,avg_latency_ms,evaluation_timestamp\n"
    row = (
        f"{metrics['label_accuracy']},"
        f"{metrics['groundedness_rate']},"
        f"{metrics['quote_integrity_rate']},"
        f"{metrics['avg_latency_ms']},"
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}Z\n"
    )
    path.write_text(header + row, encoding="utf-8")


def write_combined_csv(
    ms3_metrics: Dict[str, Any],
    ms1_csv_path: Optional[Path],
    out_path: Path,
) -> None:
    """
    Required by MS3 §5b: ONE final evaluation CSV that includes both MS1 and
    MS3 metrics. We reuse the existing MS1 CSV verbatim if it's available,
    otherwise the combined file just contains the MS3 row with no MS1 row.
    """
    header = "run,label_accuracy,groundedness,quote_integrity_pass_rate,avg_latency_ms,evaluation_timestamp\n"
    lines = [header]

    if ms1_csv_path and ms1_csv_path.exists():
        ms1_lines = ms1_csv_path.read_text(encoding="utf-8").splitlines()
        # Skip MS1 header (its first line) and prepend "ms1" to each data row
        for raw in ms1_lines[1:]:
            if raw.strip():
                lines.append(f"ms1,{raw}\n")

    ts = datetime.now(timezone.utc).isoformat(timespec='seconds') + "Z"
    lines.append(
        f"ms3,"
        f"{ms3_metrics['label_accuracy']},"
        f"{ms3_metrics['groundedness_rate']},"
        f"{ms3_metrics['quote_integrity_rate']},"
        f"{ms3_metrics['avg_latency_ms']},"
        f"{ts}\n"
    )
    out_path.write_text("".join(lines), encoding="utf-8")


def write_runtrace(
    contract_id: str,
    contract_text: str,
    agent_traces: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    verdicts: List[Dict[str, Any]],
    retrieval_mode: str,
    path: Path,
) -> None:
    """
    Minimal MS3 runtrace until Task 4 (RuntraceFormatter) lands. Includes the
    full agent_traces + tool_calls list so the formatter can rebuild a
    schema-compliant runtrace from these files later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version":  "1.0-ms3-draft",
        "contract_id":     contract_id,
        "contract_chars":  len(contract_text),
        "retrieval_mode":  retrieval_mode,
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "verdicts":        verdicts,
        "agent_traces":    agent_traces,
        "tool_calls":      tool_calls,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ── main runner ───────────────────────────────────────────────────────────────

def run_evaluation(
    orchestrator: Any,
    contracts: List[Dict[str, Any]],
    output_dir: Path,
    *,
    limit: Optional[int] = None,
    progress_cb: Optional[Any] = None,
    playbook_path: Optional[Path] = None,
    ms1_csv_path: Optional[Path] = None,
    resume: bool = True,
    shard_index: int = 0,
    shard_total: int = 1,
) -> Dict[str, Any]:
    """
    Loop over the test split, run the hypothesis pipeline on each contract,
    write per-contract runtraces, and return aggregate metrics.

    Checkpointing
    -------------
    After each contract is processed the function writes a checkpoint to
    `<output_dir>/checkpoint.json` containing the per-contract verdicts +
    accumulated per-verdict scores + latencies. If `resume=True` (the default)
    and the checkpoint already exists at startup, contracts already listed
    there are skipped on re-run. This makes the run safe to restart after a
    Kaggle kernel timeout: re-execute the cell and it picks up where it
    stopped.

    Sharding (parallel runs across machines)
    ----------------------------------------
    Pass `shard_index=K, shard_total=N` to process only the K-th contiguous
    block of `N` evenly-sized partitions. e.g. with 123 contracts and
    `shard_total=5`:
        shard 0 → contracts[0:24]
        shard 1 → contracts[24:49]
        ...
        shard 4 → contracts[98:123]
    After all shards finish, `scripts/merge_shards.py` combines the per-shard
    output directories into the final §5b combined CSV and §5c runtraces zip.

    Args:
        orchestrator:  a built Orchestrator (its hypothesis_pipeline is used).
        contracts:     list of normalised contract dicts (from get_test_contracts).
        output_dir:    directory to write predictions / runtraces / metrics.
        limit:         optional cap for smoke testing — process only the first N.
        progress_cb:   optional callable invoked as progress_cb(i, total, contract_id).
        playbook_path: path to playbook.yaml (§3c — deterministic policy mapping).
        ms1_csv_path:  path to existing MS1 metrics CSV (§5b — combined CSV).
        resume:        if True and a checkpoint exists, skip contracts already done.
        shard_index:   0-indexed shard number for parallel runs (default 0).
        shard_total:   total number of shards (default 1 = no sharding).

    Returns:
        Aggregate metrics dict (also written to evaluation_metrics_ms3.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runtrace_dir = output_dir / "runtraces"
    runtrace_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    playbook = load_playbook(playbook_path) if playbook_path else {}

    # Task 4 is wired directly into the Orchestrator on the merged branch:
    # `orchestrator.run(contract, "analyze this contract", history)` returns a
    # result dict that already contains `enriched_verdicts` (PlaybookEnricher
    # output) and `runtrace` (schema-compliant RuntraceFormatter output) when
    # the intent routes to hypothesis_analysis. We use those directly and only
    # fall back to local shims if the orchestrator didn't produce them
    # (e.g. older orchestrator without Task 4 wiring).
    has_task4_via_orchestrator = (
        hasattr(orchestrator, "enricher") and hasattr(orchestrator, "formatter")
    )
    if has_task4_via_orchestrator:
        print("[evaluate_ms3] using orchestrator's Task 4 wiring (enrichment + runtrace)", file=sys.stderr)

    # Fallback local instances (only constructed if needed for older orchestrators)
    task4_enricher = None
    if _HAS_TASK4_ENRICHER and playbook_path is not None and not has_task4_via_orchestrator:
        try:
            task4_enricher = PlaybookEnricher(str(playbook_path))  # type: ignore[misc]
            print("[evaluate_ms3] using standalone Task 4 PlaybookEnricher", file=sys.stderr)
        except Exception as exc:
            print(f"[evaluate_ms3] Task 4 PlaybookEnricher init failed ({exc}); using local shim", file=sys.stderr)

    # ── shard slicing (parallel runs across machines) ─────────────────────────
    if shard_total > 1:
        if not (0 <= shard_index < shard_total):
            raise ValueError(
                f"shard_index ({shard_index}) must be in [0, {shard_total})"
            )
        n = len(contracts)
        start = (shard_index * n) // shard_total
        end   = ((shard_index + 1) * n) // shard_total
        contracts = contracts[start:end]
        print(
            f"[evaluate_ms3] shard {shard_index + 1}/{shard_total}: "
            f"processing contracts[{start}:{end}] ({len(contracts)} of {n})",
            file=sys.stderr,
        )

    selected = contracts[: limit] if limit else contracts
    total = len(selected)

    # ── checkpoint load ───────────────────────────────────────────────────────
    # done_contracts: contract_id -> {"latency_ms": float, "verdicts": List[Dict]}
    done_contracts: Dict[str, Dict[str, Any]] = {}
    per_verdict_scores: List[Dict[str, Any]] = []
    latencies: List[float] = []
    skipped: List[str] = []

    if resume and checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            done_contracts     = cp.get("done", {})
            per_verdict_scores = cp.get("per_verdict_scores", [])
            latencies          = cp.get("latencies", [])
            skipped            = cp.get("skipped", [])
            print(
                f"[evaluate_ms3] resuming from checkpoint: "
                f"{len(done_contracts)} contracts already processed, "
                f"{len(skipped)} previously skipped",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[evaluate_ms3] checkpoint load failed ({exc}); starting fresh",
                file=sys.stderr,
            )
            done_contracts, per_verdict_scores, latencies, skipped = {}, [], [], []

    def _save_checkpoint() -> None:
        """Atomic write so an interrupted save can't corrupt the file."""
        tmp = checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({
                "done":               done_contracts,
                "per_verdict_scores": per_verdict_scores,
                "latencies":          latencies,
                "skipped":            skipped,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(checkpoint_path)

    retrieval_mode = getattr(orchestrator.retriever, "mode", "unknown")

    # Lazy import — ConversationHistory only needed when going through orchestrator.run
    if has_task4_via_orchestrator:
        from bettercallnli.agent.history import ConversationHistory  # type: ignore

    for i, contract in enumerate(selected, 1):
        c_id = contract.get("id", f"contract-{i}")

        # Skip contracts already completed (resume path)
        if c_id in done_contracts:
            if progress_cb:
                progress_cb(i, total, c_id, status="cached",
                            latency_ms=done_contracts[c_id].get("latency_ms", 0.0))
            continue

        gold_labels = extract_gold_labels(contract)
        if not gold_labels:
            if c_id not in skipped:
                skipped.append(c_id)
                _save_checkpoint()
            if progress_cb:
                progress_cb(i, total, c_id, status="skipped")
            continue

        t_start = time.perf_counter()
        try:
            if has_task4_via_orchestrator:
                # Drive the full agentic pipeline through the orchestrator so
                # PlaybookEnricher + RuntraceFormatter run automatically. The
                # IntentRouter sees "analyze this contract" and hits the
                # keyword fast-path (no LLM call), so this costs zero extra.
                if hasattr(orchestrator, "reset_session"):
                    orchestrator.reset_session()
                result = orchestrator.run(
                    contract     = contract,
                    user_message = "analyze this contract",
                    history      = ConversationHistory(),
                )
            else:
                # Older orchestrator without Task 4 wiring — call pipeline directly
                result = orchestrator.hypothesis_pipeline.run(contract)
        except Exception as exc:
            print(f"[evaluate_ms3] contract {c_id} failed: {exc}", file=sys.stderr)
            skipped.append(c_id)
            if progress_cb:
                progress_cb(i, total, c_id, status="error")
            continue
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        latencies.append(latency_ms)

        raw_verdicts: List[Dict[str, Any]] = result.get("verdicts", [])
        agent_traces: List[Dict[str, Any]] = result.get("agent_traces", [])
        tool_calls:   List[Dict[str, Any]] = result.get("tool_calls", [])

        # §3c: enrich every verdict with deterministic playbook policy fields.
        # Priority:
        #   1. result["enriched_verdicts"] from orchestrator's PlaybookEnricher (preferred)
        #   2. standalone task4_enricher.enrich() (if orchestrator didn't wire it)
        #   3. local apply_playbook() shim (last resort)
        enriched_from_orchestrator = result.get("enriched_verdicts")
        if enriched_from_orchestrator:
            verdicts = enriched_from_orchestrator
        elif task4_enricher is not None:
            try:
                verdicts = task4_enricher.enrich(raw_verdicts)
            except Exception as exc:
                print(f"[evaluate_ms3] Task 4 enrich() failed ({exc}); falling back to local shim", file=sys.stderr)
                verdicts = [apply_playbook(v, playbook) for v in raw_verdicts] if playbook else raw_verdicts
        elif playbook:
            verdicts = [apply_playbook(v, playbook) for v in raw_verdicts]
        else:
            verdicts = raw_verdicts

        for v in verdicts:
            h_id = v.get("hypothesis_id", "?")
            gold = gold_labels.get(h_id)
            if gold is None:
                continue
            score = score_verdict(v, gold, contract.get("text", ""))
            score["contract_id"]   = c_id
            score["hypothesis_id"] = h_id
            per_verdict_scores.append(score)

        done_contracts[c_id] = {
            "latency_ms": latency_ms,
            "verdicts":   verdicts,
        }

        # §2c + §2h: write the schema-compliant runtrace.
        # Priority:
        #   1. result["runtrace"] — schema-compliant payload from RuntraceFormatter
        #   2. local write_runtrace() draft (only when orchestrator didn't emit one)
        runtrace_path = runtrace_dir / f"runtrace_{c_id}.json"
        runtrace_payload = result.get("runtrace")
        if runtrace_payload and _HAS_TASK4_FORMATTER:
            try:
                RuntraceFormatter.save(runtrace_payload, str(runtrace_path))  # type: ignore[union-attr]
            except Exception as exc:
                print(f"[evaluate_ms3] RuntraceFormatter.save failed ({exc}); falling back to direct JSON write", file=sys.stderr)
                runtrace_path.write_text(json.dumps(runtrace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        elif runtrace_payload:
            runtrace_path.write_text(json.dumps(runtrace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            # §2h: every tool_call must have name, args, output, count
            normalized_tcs = normalize_tool_calls(tool_calls)
            normalized_traces = [
                {**t, "tool_calls": normalize_tool_calls(t.get("tool_calls", []))}
                for t in agent_traces
            ]
            write_runtrace(
                contract_id=c_id,
                contract_text=contract.get("text", ""),
                agent_traces=normalized_traces,
                tool_calls=normalized_tcs,
                verdicts=verdicts,
                retrieval_mode=retrieval_mode,
                path=runtrace_path,
            )

        # Persist checkpoint after each contract so a kernel timeout doesn't
        # waste work. Atomic write — safe to interrupt at any moment.
        _save_checkpoint()

        if progress_cb:
            progress_cb(i, total, c_id, status="ok", latency_ms=latency_ms)

    # ── aggregate ─────────────────────────────────────────────────────────────
    metrics = aggregate_metrics(per_verdict_scores, latencies)
    metrics["skipped"] = skipped
    metrics["retrieval_mode"] = retrieval_mode

    # Rebuild all_predictions from the checkpoint dict (resilient to partial runs).
    # Preserves the original contract-input order so the output JSON is stable.
    contract_id_order = [c.get("id", f"contract-{i}") for i, c in enumerate(selected, 1)]
    all_predictions = [
        {
            "contract_id": c_id,
            "latency_ms":  done_contracts[c_id]["latency_ms"],
            "verdicts":    done_contracts[c_id]["verdicts"],
        }
        for c_id in contract_id_order
        if c_id in done_contracts
    ]

    (output_dir / "predictions_ms3.json").write_text(
        json.dumps(all_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "evaluation_metrics_ms3.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(metrics, output_dir / "evaluation_metrics_ms3.csv")

    # §5b — ONE final CSV with both MS1 and MS3 metrics
    write_combined_csv(
        ms3_metrics=metrics,
        ms1_csv_path=ms1_csv_path,
        out_path=output_dir / "evaluation_metrics_combined.csv",
    )

    # §5c — zip every runtrace for the deliverable
    zip_base = output_dir / "runtraces_ms3"
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=runtrace_dir)
    metrics["runtrace_archive"] = str(Path(archive).name)

    return metrics


# ── CLI entry ─────────────────────────────────────────────────────────────────

def _main() -> int:
    """Standalone entry: `python -m scripts.evaluate_ms3 …`"""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    # add src/ and project root to sys.path so package + sibling-script
    # imports work when run as a standalone script
    _repo_root = Path(__file__).resolve().parent.parent
    for _p in (_repo_root / "src", _repo_root):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    from bettercallnli.agent.orchestrator import build_orchestrator
    from bettercallnli.utils.contract_loader import get_test_contracts

    parser = argparse.ArgumentParser(prog="evaluate_ms3")
    parser.add_argument("--data-dir",   default=None, help="Local ContractNLI directory (defaults to kagglehub)")
    parser.add_argument("--retrieval",  choices=["vector", "graphrag"], default="graphrag")
    parser.add_argument("--output-dir", default="results/ms3")
    parser.add_argument("--limit",      type=int, default=None, help="Process only first N contracts (smoke test)")
    parser.add_argument("--playbook",   default="playbook.yaml", help="Playbook YAML for §3c deterministic policy mapping")
    parser.add_argument("--ms1-csv",    default="results/ms1/evaluation_metrics.csv", help="Existing MS1 CSV to include in the combined CSV (§5b)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="0-indexed shard for parallel runs (default 0)")
    parser.add_argument("--shard-total", type=int, default=1,
                        help="Total number of shards (default 1 = no sharding)")
    args = parser.parse_args()

    print(f"[evaluate_ms3] retrieval mode: {args.retrieval}")
    print(f"[evaluate_ms3] output dir    : {args.output_dir}")
    print(f"[evaluate_ms3] playbook      : {args.playbook}")
    print(f"[evaluate_ms3] MS1 CSV       : {args.ms1_csv}")

    orchestrator = build_orchestrator(retrieval_mode=args.retrieval, playbook_path=args.playbook)
    contracts = get_test_contracts(local_path=args.data_dir)
    print(f"[evaluate_ms3] loaded {len(contracts)} test contracts")

    def _progress(i: int, total: int, c_id: str, *, status: str = "ok", latency_ms: float = 0.0) -> None:
        suffix = f"  ({latency_ms/1000:.1f}s)" if status == "ok" else f"  [{status}]"
        print(f"  [{i:>3}/{total}] {c_id}{suffix}")

    metrics = run_evaluation(
        orchestrator=orchestrator,
        contracts=contracts,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        progress_cb=_progress,
        playbook_path=Path(args.playbook) if args.playbook else None,
        ms1_csv_path=Path(args.ms1_csv) if args.ms1_csv else None,
        shard_index=args.shard_index,
        shard_total=args.shard_total,
    )

    print("\n=== Aggregate metrics ===")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "per_hypothesis"},
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
