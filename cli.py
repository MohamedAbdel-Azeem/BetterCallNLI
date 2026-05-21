"""
BetterCallNLI CLI — Milestone 3 (Task 5)
=========================================

Claude Code-style terminal interface for the NDA review agent. Replaces the
Streamlit-only flow with three modes:

    converse   multi-turn chat with one NDA contract
    analyze    one-shot full 17-hypothesis review of a contract
    evaluate   batch evaluation on the ContractNLI test split (MS1-comparable metrics)

Usage
-----
    python cli.py --mode converse --contract path/to/contract.txt --retrieval graphrag
    python cli.py --mode analyze  --contract path/to/contract.txt --retrieval vector
    python cli.py --mode evaluate --data-dir path/to/contractnli/ --retrieval graphrag

Environment (.env)
------------------
    HF_TOKEN           HuggingFace API token (required for HF Serverless backend)
    NEO4J_URI / _USERNAME / _PASSWORD     (required for --retrieval graphrag)
    CHROMA_API_KEY                        (required for --retrieval vector)

For Kaggle runs against the fine-tuned LoRA model, use `notebooks/evaluate_ms3.ipynb`
which installs a `LocalInferenceClient` before constructing the orchestrator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 so Windows legacy consoles (cp1252) don't choke on ✓ / ─ / ╭ etc.
# Must happen before rich is imported anywhere downstream.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Repo root needs to be importable when invoked as `python cli.py`
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from src.agent.history import ConversationHistory
from src.agent.orchestrator import build_orchestrator
from src.ui.console import (
    BRAND_ACCENT,
    BRAND_MUTED,
    BRAND_PRIMARY,
    ERR_STYLE,
    LABEL_COLOURS,
    OK_STYLE,
    WARN_STYLE,
    get_console,
    render_banner,
    render_conversation_result,
    render_hypothesis_summary,
    render_tool_calls,
    render_verdict_card,
    status,
)
from src.utils.contract_loader import (
    contract_from_text,
    get_contract_by_id,
    get_test_contracts,
)


VERSION = "0.3.0"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_contract(arg: Optional[str], data_dir: Optional[str]) -> Dict[str, Any]:
    """
    Resolve --contract into a contract dict.

    Accepts:
      - a path to a .txt file
      - a ContractNLI test-set ID (looked up via get_contract_by_id)
    """
    con = get_console()
    if not arg:
        con.print(f"[{ERR_STYLE}]✗[/] --contract is required for this mode")
        raise SystemExit(2)

    p = Path(arg)
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="ignore")
        return contract_from_text(text, contract_id=p.stem)

    # Try test-set ID
    try:
        contract = get_contract_by_id(arg, local_path=data_dir)
    except Exception as exc:
        con.print(f"[{ERR_STYLE}]✗[/] could not load test contracts: {exc}")
        raise SystemExit(2)

    if contract is None:
        con.print(
            f"[{ERR_STYLE}]✗[/] '{arg}' is neither a file nor a test-set contract ID"
        )
        raise SystemExit(2)
    return contract


def _print_contract_info(contract: Dict[str, Any], retrieval_mode: str) -> None:
    con = get_console()
    con.print(
        f"[{BRAND_MUTED}]contract[/] [bold]{contract.get('id', '?')}[/]"
        f"  [{BRAND_MUTED}]·[/]  [{BRAND_ACCENT}]{contract.get('char_count', len(contract.get('text', ''))):,} chars[/]"
        f"  [{BRAND_MUTED}]·[/]  retrieval [{BRAND_ACCENT}]{retrieval_mode}[/]"
    )


def _check_env(retrieval_mode: str) -> None:
    """Fail fast with a friendly message if required env vars are missing."""
    con = get_console()
    missing: List[str] = []
    if not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN")
    if retrieval_mode == "graphrag":
        for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
            if not os.getenv(k):
                missing.append(k)
    if retrieval_mode == "vector" and not os.getenv("CHROMA_API_KEY"):
        missing.append("CHROMA_API_KEY")

    if missing:
        con.print(
            f"[{ERR_STYLE}]✗[/] missing required environment variables: "
            + ", ".join(missing)
        )
        con.print(f"[{BRAND_MUTED}]  add them to your .env file and try again[/]")
        raise SystemExit(2)


# ── mode: converse ────────────────────────────────────────────────────────────

def run_converse(args: argparse.Namespace) -> int:
    con = get_console()
    render_banner(version=VERSION, subtitle="NDA Review Agent · converse")

    _check_env(args.retrieval)

    with status("Building orchestrator", ok_message=f"Orchestrator ready · {args.retrieval}"):
        orchestrator = build_orchestrator(
            retrieval_mode=args.retrieval,
            playbook_path=args.playbook,
        )

    contract = _load_contract(args.contract, args.data_dir)
    _print_contract_info(contract, args.retrieval)

    history = ConversationHistory()
    con.print(
        f"\n[{BRAND_MUTED}]session[/] [bold]{history.session_id}[/]  "
        f"[{BRAND_MUTED}](type 'exit' to quit, 'reset' to clear history)[/]\n"
    )

    # §2c: accumulate every turn into a per-session runtrace
    session_turns: List[Dict[str, Any]] = []

    while True:
        try:
            user_input = con.input(f"[{BRAND_PRIMARY}]You ›[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            con.print("\n[grey62]session ended[/]")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            con.print("[grey62]session ended[/]")
            break
        if user_input.lower() == "reset":
            history.clear()
            session_turns.clear()
            con.print(f"[{OK_STYLE}]✓[/] history cleared")
            continue

        try:
            with status("Routing intent", ok_message="Intent resolved"):
                result = orchestrator.run(
                    contract=contract,
                    user_message=user_input,
                    history=history,
                )
        except NotImplementedError as exc:
            con.print(f"[{WARN_STYLE}]![/] {exc}")
            continue
        except Exception as exc:
            con.print(f"[{ERR_STYLE}]✗[/] orchestrator error: {exc}")
            continue

        mode = result.get("mode", "conversation")
        con.print(f"[{BRAND_MUTED}]→ mode[/] [bold]{mode}[/]")

        if mode == "conversation":
            render_conversation_result(result)
        else:
            verdicts = result.get("verdicts", []) or []
            render_hypothesis_summary(verdicts, title=f"Verdicts for {contract['id']}")

        if args.verbose and result.get("tool_calls"):
            render_tool_calls(result["tool_calls"])

        session_turns.append({
            "turn_id":      len(session_turns) + 1,
            "user_message": user_input,
            "intent":       mode,
            "response":     result.get("response", ""),
            "evidence":     result.get("evidence", []),
            "precedents":   result.get("precedents", []),
            "verdicts":     result.get("verdicts", []),
            "tool_calls":   _normalize_tool_calls(result.get("tool_calls", [])),
            "retrieval_mode": result.get("retrieval_mode", args.retrieval),
        })

        if args.save_history:
            history.save(args.save_history)

    if args.save_history:
        con.print(f"[{OK_STYLE}]✓[/] history saved to {args.save_history}")

    # §2c: write the per-session conversation runtrace.
    # Prefer the orchestrator's RuntraceFormatter-backed builder (schema-compliant);
    # fall back to the local draft writer if the orchestrator doesn't expose it.
    session_runtrace_path = (
        Path(args.session_runtrace) if args.session_runtrace
        else Path("results/ms3/conversation_runtraces") / f"session_{history.session_id}.json"
    )
    wrote_session_runtrace = False
    if hasattr(orchestrator, "build_session_runtrace") and len(history) > 0:
        try:
            payload = orchestrator.build_session_runtrace(contract=contract, history=history)
            session_runtrace_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.utils.runtrace import RuntraceFormatter  # type: ignore
                RuntraceFormatter.save(payload, str(session_runtrace_path))
            except Exception:
                session_runtrace_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            con.print(f"[{OK_STYLE}]✓[/] session runtrace written to [bold]{session_runtrace_path}[/]")
            wrote_session_runtrace = True
        except Exception as exc:
            con.print(f"[{WARN_STYLE}]![/] orchestrator session runtrace failed ({exc}); using local fallback")

    if not wrote_session_runtrace:
        _write_conversation_runtrace(
            session_id=history.session_id,
            contract=contract,
            retrieval_mode=args.retrieval,
            turns=session_turns,
            output_path=session_runtrace_path,
        )

    return 0


def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """§2h — every tool_call must carry {name, args, output, count}."""
    out = []
    for tc in tool_calls or []:
        out.append({
            "name":   tc.get("name", "unknown"),
            "args":   tc.get("args", {}),
            "output": tc.get("output", {}),
            "count":  int(tc.get("count", 1)),
            **({"latency_ms": tc["latency_ms"]} if "latency_ms" in tc else {}),
        })
    return out


def _write_conversation_runtrace(
    session_id: str,
    contract: Dict[str, Any],
    retrieval_mode: str,
    turns: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """§2c — one runtrace per conversation session."""
    if not turns:
        return
    from datetime import datetime, timezone
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version":  "1.0-ms3-draft",
        "kind":            "conversation_session",
        "session_id":      session_id,
        "contract_id":     contract.get("id"),
        "retrieval_mode":  retrieval_mode,
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "turn_count":      len(turns),
        "turns":           turns,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    get_console().print(
        f"[{OK_STYLE}]✓[/] session runtrace written to [bold]{output_path}[/]"
    )


# ── mode: analyze ─────────────────────────────────────────────────────────────

def run_analyze(args: argparse.Namespace) -> int:
    con = get_console()
    render_banner(version=VERSION, subtitle="NDA Review Agent · analyze")

    _check_env(args.retrieval)

    with status("Building orchestrator", ok_message=f"Orchestrator ready · {args.retrieval}"):
        orchestrator = build_orchestrator(
            retrieval_mode=args.retrieval,
            playbook_path=args.playbook,
        )

    contract = _load_contract(args.contract, args.data_dir)
    _print_contract_info(contract, args.retrieval)

    con.print(f"\n[{BRAND_PRIMARY}]→[/] running 17-hypothesis analysis…\n")
    t0 = time.perf_counter()
    try:
        result = orchestrator.run(
            contract=contract,
            user_message="analyze this contract",
            history=ConversationHistory(),
        )
    except NotImplementedError as exc:
        con.print(f"[{WARN_STYLE}]![/] {exc}")
        return 1
    elapsed = time.perf_counter() - t0

    raw_verdicts = result.get("verdicts", []) or []

    # §3c: prefer the orchestrator's enriched_verdicts (PlaybookEnricher output).
    # Fall back to the local apply_playbook shim if the orchestrator didn't emit them.
    enriched_from_orch = result.get("enriched_verdicts")
    if enriched_from_orch:
        verdicts = enriched_from_orch
    else:
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from scripts.evaluate_ms3 import apply_playbook, load_playbook  # type: ignore
        try:
            playbook = load_playbook(Path(args.playbook))
            verdicts = [apply_playbook(v, playbook) for v in raw_verdicts]
        except Exception as exc:
            con.print(f"[{WARN_STYLE}]![/] playbook enrichment skipped: {exc}")
            verdicts = raw_verdicts

    con.print(f"[{OK_STYLE}]✓[/] analysis complete in {elapsed:.1f}s · {len(verdicts)} verdicts")

    render_hypothesis_summary(verdicts, title=f"Verdicts for {contract['id']}")

    if args.show_cards:
        for v in verdicts:
            render_verdict_card(v)

    if args.verbose and result.get("tool_calls"):
        render_tool_calls(result["tool_calls"])

    if args.output:
        payload = {
            "contract_id": contract["id"],
            "mode":        result.get("mode"),
            "verdicts":    verdicts,
            "tool_calls":  result.get("tool_calls", []),
        }
        # If the orchestrator built a schema-compliant runtrace, include it
        if result.get("runtrace"):
            payload["runtrace"] = result["runtrace"]
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        con.print(f"[{OK_STYLE}]✓[/] result written to {args.output}")

    return 0


# ── mode: evaluate ────────────────────────────────────────────────────────────

def run_evaluate(args: argparse.Namespace) -> int:
    con = get_console()
    render_banner(version=VERSION, subtitle="NDA Review Agent · evaluate")

    _check_env(args.retrieval)

    # Import the runner lazily — pulls in dataset / metrics helpers
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from scripts.evaluate_ms3 import run_evaluation

    with status("Building orchestrator", ok_message=f"Orchestrator ready · {args.retrieval}"):
        orchestrator = build_orchestrator(
            retrieval_mode=args.retrieval,
            playbook_path=args.playbook,
        )

    with status("Loading ContractNLI test split", ok_message="Test split loaded"):
        contracts = get_test_contracts(local_path=args.data_dir)
    con.print(f"[{BRAND_MUTED}]  contracts[/] [bold]{len(contracts)}[/]")

    if args.limit:
        con.print(f"[{WARN_STYLE}]![/] limiting to first {args.limit} contracts (smoke mode)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    con.print(f"[{BRAND_MUTED}]  output dir[/] {output_dir}\n")

    # rich progress bar for the eval loop
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    total = args.limit or len(contracts)
    with Progress(
        SpinnerColumn(style=BRAND_ACCENT),
        TextColumn("[bold]{task.fields[c_id]}[/]"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=con,
    ) as progress:
        task_id = progress.add_task("evaluate", total=total, c_id="—")

        def _cb(i: int, n: int, c_id: str, *, status: str = "ok", latency_ms: float = 0.0) -> None:
            progress.update(task_id, completed=i, c_id=f"{c_id} [{status}]")

        metrics = run_evaluation(
            orchestrator=orchestrator,
            contracts=contracts,
            output_dir=output_dir,
            limit=args.limit,
            progress_cb=_cb,
            playbook_path=Path(args.playbook) if args.playbook else None,
            ms1_csv_path=Path(args.ms1_csv) if args.ms1_csv else None,
        )

    # ── final summary ─────────────────────────────────────────────────────────
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.table import Table

    table = Table(
        show_header=False, box=ROUNDED, border_style=BRAND_MUTED, expand=False,
    )
    table.add_column("metric", style=BRAND_MUTED)
    table.add_column("value", style="bold")

    table.add_row("contracts",              str(metrics["contract_count"]))
    table.add_row("hypotheses (scored)",    str(metrics["hypothesis_count"]))
    table.add_row("label accuracy",         f"{metrics['label_accuracy']:.4f}")
    table.add_row("groundedness rate",      f"{metrics['groundedness_rate']:.4f}")
    table.add_row("quote integrity rate",   f"{metrics['quote_integrity_rate']:.4f}")
    table.add_row("avg latency / contract", f"{metrics['avg_latency_ms']:.0f} ms")
    table.add_row("retrieval mode",         metrics["retrieval_mode"])
    if metrics.get("skipped"):
        table.add_row("skipped contracts",  str(len(metrics["skipped"])))

    con.print()
    con.print(Panel(
        table,
        title=f"[{BRAND_PRIMARY}]Aggregate metrics[/]",
        title_align="left",
        border_style=BRAND_PRIMARY,
        box=ROUNDED,
        padding=(0, 1),
    ))

    con.print(
        f"\n[{BRAND_MUTED}]files written to[/] [bold]{output_dir}[/]\n"
        f"  · predictions_ms3.json\n"
        f"  · evaluation_metrics_ms3.csv\n"
        f"  · evaluation_metrics_ms3.json\n"
        f"  · evaluation_metrics_combined.csv   ([{BRAND_ACCENT}]MS1 + MS3, deliverable §5b[/])\n"
        f"  · runtraces/runtrace_<id>.json  ({total} files)\n"
        f"  · runtraces_ms3.zip                ([{BRAND_ACCENT}]zipped runtraces, deliverable §5c[/])"
    )

    return 0


# ── parser + dispatcher ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="BetterCallNLI",
        description="Claude Code-style CLI for the BetterCallNLI NDA review agent.",
    )
    p.add_argument(
        "--mode",
        choices=["converse", "analyze", "evaluate"],
        required=True,
    )
    p.add_argument(
        "--retrieval",
        choices=["vector", "graphrag"],
        default="graphrag",
        help="Retriever backend (default: graphrag).",
    )
    p.add_argument(
        "--contract",
        help="Path to a .txt contract OR a test-set contract ID (for converse/analyze).",
    )
    p.add_argument(
        "--data-dir",
        help="Local ContractNLI directory (defaults to kagglehub download).",
    )
    p.add_argument(
        "--playbook",
        default="playbook.yaml",
        help="Path to playbook.yaml (default: ./playbook.yaml).",
    )
    p.add_argument("--output",            help="(analyze) write final result JSON here.")
    p.add_argument("--output-dir",        default="results/ms3", help="(evaluate) output directory.")
    p.add_argument("--limit",             type=int, default=None, help="(evaluate) cap number of contracts.")
    p.add_argument("--save-history",      help="(converse) write ConversationHistory JSON on every turn.")
    p.add_argument("--session-runtrace",  help="(converse) path for the per-session runtrace (§2c).")
    p.add_argument("--ms1-csv",           default="results/evaluation_metrics.csv",
                   help="(evaluate) existing MS1 CSV to merge into the combined CSV (§5b).")
    p.add_argument("--show-cards",        action="store_true", help="(analyze) print one panel per verdict.")
    p.add_argument("-v", "--verbose",     action="store_true", help="Print tool-call traces.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.mode == "converse":
        return run_converse(args)
    if args.mode == "analyze":
        return run_analyze(args)
    if args.mode == "evaluate":
        return run_evaluate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
