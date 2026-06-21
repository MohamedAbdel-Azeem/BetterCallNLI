"""
BetterCallNLI CLI — Milestone 3 (Task 5)
=========================================

Claude Code-style terminal interface for the NDA review agent. Supports three
top-level modes, but the default interactive mode now routes intent per-prompt
via IntentRouter rather than locking the session to one mode at launch.

    interactive  default — multi-turn session; intent routed per message
    analyze      one-shot full 17-hypothesis review of a contract (non-interactive)
    evaluate     batch evaluation on the ContractNLI test split (MS1-comparable metrics)

Usage
-----
    # Interactive (recommended) — contract and intent set per prompt
    python cli.py --retrieval graphrag
    python cli.py --retrieval vector

    # Inside the interactive session:
    #   @path/to/contract.txt what are the confidentiality obligations?
    #   @1 analyze             ← loads contract ID 1, shows hypothesis picker
    #   what does clause 4 mean?          ← reuses last loaded contract
    #   analyze                            ← router picks hypothesis mode, picker shown
    #   cards on / cards off               ← toggle verdict card rendering

    # Non-interactive one-shot analysis (always runs all 17)
    python cli.py --mode analyze --contract path/to/contract.txt --retrieval graphrag

    # Batch evaluation
    python cli.py --mode evaluate --data-dir path/to/contractnli/ --retrieval graphrag

    # Legacy locked-mode converse (backwards compatible)
    python cli.py --mode converse --contract path/to/contract.txt --retrieval graphrag

Environment (.env)
------------------
    HF_TOKEN           HuggingFace API token (required)
    NEO4J_URI / _USERNAME / _PASSWORD     (required for --retrieval graphrag)
    CHROMA_API_KEY                        (required for --retrieval vector)

For Kaggle runs against the fine-tuned LoRA model, use `notebooks/evaluate_ms3.ipynb`
which installs a `LocalInferenceClient` before constructing the orchestrator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 so Windows legacy consoles (cp1252) don't choke on ✓ / ─ / ╭ etc.
# Must happen before rich is imported anywhere downstream.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Make the package (src/) and the repo root importable when run as
# `python apps/cli.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv()

from bettercallnli.agent.history import ConversationHistory
from bettercallnli.agent.orchestrator import build_orchestrator
from bettercallnli.ui.console import (
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
from bettercallnli.utils.contract_loader import (
    contract_from_text,
    get_contract_by_id,
    get_test_contracts,
)


VERSION = "0.3.0"

# ── ContractNLI hypotheses (from playbook.yaml) ───────────────────────────────
# Each entry: (hypothesis_id, title, hypothesis_text)
_HYPOTHESES: List[Tuple[str, str, str]] = [
    ("H01", "Explicit identification",
     "All Confidential Information shall be expressly identified by the Disclosing Party."),
    ("H02", "Technical-only scope",
     "Confidential Information shall only include technical information."),
    ("H03", "Verbal information included",
     "Confidential Information may include verbally conveyed information."),
    ("H04", "Purpose limitation",
     "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement."),
    ("H05", "Disclosure to employees",
     "Receiving Party may share some Confidential Information with some of Receiving Party's employees."),
    ("H06", "Disclosure to third parties / representatives",
     "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors)."),
    ("H07", "Notice for compelled disclosure",
     "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."),
    ("H08", "Confidentiality of the agreement/existence",
     "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated."),
    ("H09", "No reverse engineering",
     "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information."),
    ("H10", "Independent development permitted",
     "Receiving Party may independently develop information similar to Confidential Information."),
    ("H11", "Third-party acquisition permitted",
     "Receiving Party may acquire information similar to Confidential Information from a third party."),
    ("H12", "No rights / no license",
     "Agreement shall not grant Receiving Party any right to Confidential Information."),
    ("H13", "Return or destruction on termination",
     "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."),
    ("H14", "Copying permitted in some circumstances",
     "Receiving Party may create a copy of some Confidential Information in some circumstances."),
    ("H15", "Non-solicitation",
     "Receiving Party shall not solicit some of Disclosing Party's representatives."),
    ("H16", "Survival of obligations",
     "Some obligations of Agreement may survive termination of Agreement."),
    ("H17", "Retention after return/destruction",
     "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information."),
]

# ── internal commands available during interactive sessions ───────────────────
_INTERNAL_CMDS = {
    "exit", "quit", "q",       # end session
    "reset",                    # clear history
    "cards on", "cards off",    # toggle verdict card rendering
    "contract",                 # show currently loaded contract info
    "help",                     # print available commands
}

_HELP_TEXT = """
[bold]Interactive session commands[/bold]

  [bold cyan]@path/to/file.txt[/] [dim]text…[/]   load a contract from a file path
  [bold cyan]@contract-id[/] [dim]text…[/]         load a contract by test-set ID (e.g. @1, @2)
  [bold cyan]analyze[/]                             trigger hypothesis analysis with picker
  [bold cyan]cards on[/] / [bold cyan]cards off[/]            toggle full verdict card rendering
  [bold cyan]contract[/]                            show currently loaded contract info
  [bold cyan]reset[/]                               clear conversation history
  [bold cyan]exit[/] / [bold cyan]quit[/] / [bold cyan]q[/]           end the session

Any other input is sent to the agent. The IntentRouter decides whether it
becomes a conversation turn or a hypothesis-analysis run automatically.
When hypothesis_analysis is detected, a picker lets you choose which to run.
"""


# ── hypothesis picker ─────────────────────────────────────────────────────────

def _select_hypotheses() -> List[Tuple[str, str, str]]:
    """
    Interactive list picker for the 17 ContractNLI hypotheses.

    Displays a numbered menu and lets the user pick a subset by number,
    a range, or 'all'. Returns a list of (id, title, text) tuples for
    the selected hypotheses.

    The user can enter:
        all          → all 17
        1,3,5        → H01, H03, H05
        1-5          → H01 through H05
        1-3,7,10-12  → mixed ranges and singles
    """
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.table import Table

    con = get_console()

    # Build display table
    table = Table(
        show_header=True,
        header_style=BRAND_ACCENT,
        box=ROUNDED,
        border_style=BRAND_MUTED,
        expand=False,
    )
    table.add_column("#",    justify="right", style=BRAND_MUTED, width=3)
    table.add_column("ID",   style="bold",    width=4)
    table.add_column("Title",                 width=36)
    table.add_column("Hypothesis text",       overflow="fold")

    for i, (h_id, title, text) in enumerate(_HYPOTHESES, 1):
        table.add_row(str(i), h_id, title, text)

    con.print()
    con.print(Panel(
        table,
        title=f"[{BRAND_PRIMARY}]Select hypotheses to analyze[/]",
        title_align="left",
        border_style=BRAND_PRIMARY,
        box=ROUNDED,
        padding=(0, 1),
    ))
    con.print(
        f"[{BRAND_MUTED}]Enter numbers, ranges, or 'all'  "
        f"(e.g. [bold]all[/bold]  ·  [bold]1,3,5[/bold]  ·  [bold]1-5[/bold]  ·  [bold]1-3,7,10-12[/bold])[/]"
    )

    while True:
        try:
            raw = con.input(f"[{BRAND_PRIMARY}]Hypotheses ›[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            con.print(f"\n[{BRAND_MUTED}]Selection cancelled — running all 17[/]")
            return list(_HYPOTHESES)

        if not raw:
            continue

        if raw.lower() == "all":
            con.print(f"[{OK_STYLE}]✓[/] running all 17 hypotheses")
            return list(_HYPOTHESES)

        # Parse numbers and ranges
        selected_indices: List[int] = []
        valid = True
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                bounds = part.split("-", 1)
                if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                    lo, hi = int(bounds[0]), int(bounds[1])
                    if 1 <= lo <= hi <= 17:
                        selected_indices.extend(range(lo, hi + 1))
                    else:
                        con.print(f"[{ERR_STYLE}]✗[/] range '{part}' out of bounds (1–17)")
                        valid = False
                        break
                else:
                    con.print(f"[{ERR_STYLE}]✗[/] invalid range '{part}'")
                    valid = False
                    break
            elif part.isdigit():
                n = int(part)
                if 1 <= n <= 17:
                    selected_indices.append(n)
                else:
                    con.print(f"[{ERR_STYLE}]✗[/] '{part}' out of bounds (1–17)")
                    valid = False
                    break
            else:
                con.print(f"[{ERR_STYLE}]✗[/] unrecognised token '{part}' — use numbers, ranges, or 'all'")
                valid = False
                break

        if not valid:
            continue

        # Deduplicate and preserve order
        seen = set()
        unique = []
        for idx in selected_indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)

        if not unique:
            con.print(f"[{WARN_STYLE}]![/] no valid selections — try again")
            continue

        chosen = [_HYPOTHESES[i - 1] for i in unique]
        ids = ", ".join(h[0] for h in chosen)
        con.print(f"[{OK_STYLE}]✓[/] selected {len(chosen)} hypothes{'is' if len(chosen)==1 else 'es'}: [bold]{ids}[/bold]")
        return chosen


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_contract_from_ref(ref: str, data_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load a contract from a file path or test-set ID string.
    Returns None and prints an error if loading fails (does NOT raise SystemExit).
    """
    con = get_console()
    p = Path(ref)
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="ignore")
        return contract_from_text(text, contract_id=p.stem)

    # Try test-set ID lookup — coerce to string so numeric IDs (int in dataset) match
    try:
        contract = get_contract_by_id(ref, local_path=data_dir)
    except Exception as exc:
        con.print(f"[{ERR_STYLE}]✗[/] could not load contract '{ref}': {exc}")
        return None

    if contract is None:
        con.print(
            f"[{ERR_STYLE}]✗[/] '{ref}' is neither a valid file path nor a test-set contract ID. "
            f"(Available numeric IDs: 1, 2, 4, 5, 6, 8, 11, 18, 21, 22, …)"
        )
    return contract


def _load_contract(arg: Optional[str], data_dir: Optional[str]) -> Dict[str, Any]:
    """
    Resolve --contract CLI arg into a contract dict.
    Raises SystemExit(2) on failure (used for non-interactive modes).
    """
    con = get_console()
    if not arg:
        con.print(f"[{ERR_STYLE}]✗[/] --contract is required for this mode")
        raise SystemExit(2)

    contract = _load_contract_from_ref(arg, data_dir)
    if contract is None:
        raise SystemExit(2)
    return contract


def _parse_prompt(
    raw: str,
    data_dir: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract an @ref contract reference from a raw prompt string.

    Supports:
        @path/to/file.txt some question
        some question @nda-001
        @1

    Returns (contract_or_None, cleaned_message_without_@ref).
    If the @ref fails to resolve, returns (None, raw) so the caller can decide.
    """
    match = re.search(r"@(\S+)", raw)
    if not match:
        return None, raw

    ref = match.group(1)
    # Remove the @ref token from the message sent to the agent
    cleaned = (raw[: match.start()] + raw[match.end() :]).strip()

    contract = _load_contract_from_ref(ref, data_dir)
    return contract, cleaned


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
        "schema_version": "1.0-ms3-draft",
        "kind":           "conversation_session",
        "session_id":     session_id,
        "contract_id":    contract.get("id"),
        "retrieval_mode": retrieval_mode,
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "turn_count":     len(turns),
        "turns":          turns,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    get_console().print(
        f"[{OK_STYLE}]✓[/] session runtrace written to [bold]{output_path}[/]"
    )


def _flush_session_runtrace(
    orchestrator: Any,
    contract: Dict[str, Any],
    history: ConversationHistory,
    session_turns: List[Dict[str, Any]],
    retrieval_mode: str,
    session_runtrace_arg: Optional[str],
) -> None:
    """
    Write the per-session runtrace at end of an interactive session.
    Prefers orchestrator.build_session_runtrace(); falls back to local draft writer.
    Only writes if the session had at least one turn.
    """
    if not session_turns:
        return

    con = get_console()
    session_runtrace_path = (
        Path(session_runtrace_arg)
        if session_runtrace_arg
        else Path("results/ms3/conversation_runtraces")
        / f"session_{history.session_id}.json"
    )

    wrote = False
    if hasattr(orchestrator, "build_session_runtrace") and len(history) > 0:
        try:
            payload = orchestrator.build_session_runtrace(
                contract=contract, history=history
            )
            session_runtrace_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from bettercallnli.utils.runtrace import RuntraceFormatter  # type: ignore

                RuntraceFormatter.save(payload, str(session_runtrace_path))
            except Exception:
                session_runtrace_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            con.print(
                f"[{OK_STYLE}]✓[/] session runtrace written to [bold]{session_runtrace_path}[/]"
            )
            wrote = True
        except Exception as exc:
            con.print(
                f"[{WARN_STYLE}]![/] orchestrator session runtrace failed ({exc}); using local fallback"
            )

    if not wrote:
        _write_conversation_runtrace(
            session_id=history.session_id,
            contract=contract,
            retrieval_mode=retrieval_mode,
            turns=session_turns,
            output_path=session_runtrace_path,
        )


# ── mode: interactive (default, per-prompt intent routing) ───────────────────

def run_interactive(args: argparse.Namespace) -> int:
    """
    Default interactive session.

    - Contract is set per-prompt using @path/to/file.txt or @contract-id syntax.
    - If --contract is supplied at launch it becomes the initial contract.
    - Intent is determined per-prompt by IntentRouter (via orchestrator.run()).
    - When hypothesis_analysis intent is detected, _select_hypotheses() is called
      so the user can pick a subset before the pipeline runs.
    - --mode converse at launch locks the session to conversation intent only
      (backwards-compatible with the old converse mode).
    """
    con = get_console()
    locked_mode = args.mode  # None → router decides; "converse" → always conversation

    subtitle = (
        "NDA Review Agent · converse [dim](intent locked)[/dim]"
        if locked_mode == "converse"
        else "NDA Review Agent · interactive"
    )
    render_banner(version=VERSION, subtitle=subtitle)

    _check_env(args.retrieval)

    with status("Building orchestrator", ok_message=f"Orchestrator ready · {args.retrieval}"):
        orchestrator = build_orchestrator(
            retrieval_mode=args.retrieval,
            playbook_path=args.playbook,
        )

    # Optionally pre-load contract from --contract flag
    current_contract: Optional[Dict[str, Any]] = None
    if args.contract:
        current_contract = _load_contract(args.contract, args.data_dir)
        _print_contract_info(current_contract, args.retrieval)

    history = ConversationHistory()
    con.print(
        f"\n[{BRAND_MUTED}]session[/] [bold]{history.session_id}[/]  "
        f"[{BRAND_MUTED}](type [bold]help[/bold] for commands)[/]\n"
    )

    if current_contract is None:
        con.print(
            f"[{BRAND_MUTED}]  No contract loaded. "
            f"Start a message with [{BRAND_ACCENT}]@path/to/contract.txt[/] "
            f"or [{BRAND_ACCENT}]@contract-id[/] to load one.[/]\n"
        )

    # Per-session state
    session_turns: List[Dict[str, Any]] = []
    show_cards: bool = args.show_cards  # runtime toggle via "cards on/off"

    while True:
        # ── prompt ────────────────────────────────────────────────────────────
        contract_hint = (
            f"[{BRAND_MUTED}]({current_contract['id']})[/] "
            if current_contract
            else f"[{WARN_STYLE}](no contract)[/] "
        )
        try:
            user_input = con.input(
                f"[{BRAND_PRIMARY}]You ›[/] {contract_hint}"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            con.print("\n[grey62]session ended[/]")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # ── internal commands ─────────────────────────────────────────────────
        if lower in {"exit", "quit", "q"}:
            con.print("[grey62]session ended[/]")
            break

        if lower == "help":
            con.print(_HELP_TEXT)
            continue

        if lower == "reset":
            history.clear()
            session_turns.clear()
            orchestrator.reset_session()
            con.print(f"[{OK_STYLE}]✓[/] history and session state cleared")
            continue

        if lower == "cards on":
            show_cards = True
            con.print(f"[{OK_STYLE}]✓[/] verdict cards [bold]on[/bold]")
            continue

        if lower == "cards off":
            show_cards = False
            con.print(f"[{OK_STYLE}]✓[/] verdict cards [bold]off[/bold]")
            continue

        if lower == "contract":
            if current_contract:
                _print_contract_info(current_contract, args.retrieval)
            else:
                con.print(f"[{WARN_STYLE}]![/] no contract loaded")
            continue

        # ── parse @ref out of the message ─────────────────────────────────────
        new_contract, message = _parse_prompt(user_input, args.data_dir)

        if new_contract is not None:
            current_contract = new_contract
            con.print(
                f"[{OK_STYLE}]✓[/] contract loaded: [bold]{current_contract['id']}[/]  "
                f"[{BRAND_MUTED}]"
                f"{current_contract.get('char_count', len(current_contract.get('text', ''))):,} chars[/]"
            )

        # If the @ref failed (returns None but there WAS an @ in the input),
        # _parse_prompt already printed an error; skip this turn.
        if "@" in user_input and new_contract is None and current_contract is None:
            continue

        if not current_contract:
            con.print(
                f"[{WARN_STYLE}]![/] no contract loaded — "
                f"start your message with [{BRAND_ACCENT}]@path/to/contract.txt[/]"
            )
            continue

        # If the message is now empty after stripping the @ref, default to a
        # generic analysis trigger so the router has something to work with.
        if not message:
            message = "analyze this contract"

        # ── pre-route intent check for hypothesis picker ──────────────────────
        # We do a lightweight keyword check here (mirrors IntentRouter's own fast
        # path) so we can show the picker BEFORE the expensive pipeline call.
        # The orchestrator still calls the full router internally — this is only
        # to decide whether to show the picker, not to bypass the router.
        _hypothesis_triggers = {
            "analyze", "analyse", "analysis", "full review", "run review",
            "full analysis", "all hypotheses", "all hypothesis", "check hypotheses",
            "hypothesis analysis", "17 hypotheses", "evaluate contract", "nli",
            "contract review", "structured review", "run pipeline", "generate report",
        }
        looks_like_hypothesis = any(t in message.lower() for t in _hypothesis_triggers)

        # Show picker only in unlocked mode (not when locked to "converse")
        selected_hypotheses: Optional[List[Tuple[str, str, str]]] = None
        if looks_like_hypothesis and locked_mode != "converse":
            selected_hypotheses = _select_hypotheses()

        # ── dispatch ──────────────────────────────────────────────────────────
        try:
            with status("Thinking", ok_message="Done"):
                result = orchestrator.run(
                    contract=current_contract,
                    user_message=message,
                    history=history,
                    # Pass selected hypotheses so the pipeline can filter.
                    # Falls back to all 17 if None (orchestrator handles it).
                    **({"hypotheses": selected_hypotheses} if selected_hypotheses is not None else {}),
                )
        except NotImplementedError as exc:
            con.print(f"[{WARN_STYLE}]![/] {exc}")
            continue
        except Exception as exc:
            con.print(f"[{ERR_STYLE}]✗[/] orchestrator error: {exc}")
            continue

        detected_mode = result.get("mode", "conversation")

        # Locked-mode guard: if session is locked to converse but router picked
        # hypothesis_analysis, warn and show a summary instead of full cards.
        if locked_mode == "converse" and detected_mode == "hypothesis_analysis":
            con.print(
                f"[{WARN_STYLE}]![/] intent router detected [bold]hypothesis_analysis[/] "
                f"but session is locked to [bold]converse[/] mode. "
                f"Showing summary only — run without [bold]--mode converse[/] to unlock."
            )

        con.print(
            f"[{BRAND_MUTED}]→ intent[/] [bold]{detected_mode}[/]"
            + (f"  [{BRAND_MUTED}](locked: {locked_mode})[/]" if locked_mode else "")
        )

        # ── render result ─────────────────────────────────────────────────────
        if detected_mode == "conversation":
            render_conversation_result(result)

        else:  # hypothesis_analysis
            verdicts = result.get("verdicts", []) or []

            # Prefer enriched verdicts if the orchestrator produced them
            enriched = result.get("enriched_verdicts")
            display_verdicts = enriched if enriched else verdicts

            # Show which hypotheses were actually run
            if selected_hypotheses is not None:
                ids_run = ", ".join(h[0] for h in selected_hypotheses)
                con.print(f"[{BRAND_MUTED}]→ hypotheses run:[/] [bold]{ids_run}[/]")

            render_hypothesis_summary(
                display_verdicts,
                title=f"Verdicts for {current_contract['id']}",
            )
            if show_cards:
                for v in display_verdicts:
                    render_verdict_card(v)

        if args.verbose and result.get("tool_calls"):
            render_tool_calls(result["tool_calls"])

        # ── accumulate turn for runtrace ──────────────────────────────────────
        session_turns.append({
            "turn_id":             len(session_turns) + 1,
            "user_message":        user_input,
            "parsed_message":      message,
            "intent":              detected_mode,
            "contract_id":         current_contract.get("id"),
            "response":            result.get("response", ""),
            "evidence":            result.get("evidence", []),
            "precedents":          result.get("precedents", []),
            "verdicts":            result.get("verdicts", []),
            "tool_calls":          _normalize_tool_calls(result.get("tool_calls", [])),
            "retrieval_mode":      result.get("retrieval_mode", args.retrieval),
            # Record which hypotheses were selected (None = all 17 / conversation)
            "hypotheses_selected": (
                [h[0] for h in selected_hypotheses]
                if selected_hypotheses is not None
                else None
            ),
        })

        if args.save_history:
            history.save(args.save_history)

    # ── end of session ────────────────────────────────────────────────────────
    if args.save_history:
        con.print(f"[{OK_STYLE}]✓[/] history saved to {args.save_history}")

    if current_contract:
        _flush_session_runtrace(
            orchestrator=orchestrator,
            contract=current_contract,
            history=history,
            session_turns=session_turns,
            retrieval_mode=args.retrieval,
            session_runtrace_arg=args.session_runtrace,
        )
    else:
        con.print(f"[{BRAND_MUTED}]  no contract was loaded — skipping runtrace[/]")

    return 0


# ── mode: analyze (non-interactive one-shot, always all 17) ───────────────────

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

    # §3c: prefer orchestrator's enriched_verdicts; fall back to local shim
    enriched_from_orch = result.get("enriched_verdicts")
    if enriched_from_orch:
        verdicts = enriched_from_orch
    else:
        from scripts.evaluate_ms3 import apply_playbook, load_playbook  # type: ignore

        try:
            playbook = load_playbook(Path(args.playbook))
            verdicts = [apply_playbook(v, playbook) for v in raw_verdicts]
        except Exception as exc:
            con.print(f"[{WARN_STYLE}]![/] playbook enrichment skipped: {exc}")
            verdicts = raw_verdicts

    con.print(
        f"[{OK_STYLE}]✓[/] analysis complete in {elapsed:.1f}s · {len(verdicts)} verdicts"
    )

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
        if result.get("runtrace"):
            payload["runtrace"] = result["runtrace"]
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        con.print(f"[{OK_STYLE}]✓[/] result written to {args.output}")

    return 0


# ── mode: evaluate (batch) ────────────────────────────────────────────────────

def run_evaluate(args: argparse.Namespace) -> int:
    con = get_console()
    render_banner(version=VERSION, subtitle="NDA Review Agent · evaluate")

    _check_env(args.retrieval)

    from scripts.evaluate_ms3 import run_evaluation  # type: ignore

    with status("Building orchestrator", ok_message=f"Orchestrator ready · {args.retrieval}"):
        orchestrator = build_orchestrator(
            retrieval_mode=args.retrieval,
            playbook_path=args.playbook,
        )

    with status("Loading ContractNLI test split", ok_message="Test split loaded"):
        contracts = get_test_contracts(local_path=args.data_dir)
    con.print(f"[{BRAND_MUTED}]  contracts[/] [bold]{len(contracts)}[/]")

    if args.limit:
        con.print(
            f"[{WARN_STYLE}]![/] limiting to first {args.limit} contracts (smoke mode)"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    con.print(f"[{BRAND_MUTED}]  output dir[/] {output_dir}\n")

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

        def _cb(
            i: int,
            n: int,
            c_id: str,
            *,
            status: str = "ok",
            latency_ms: float = 0.0,
        ) -> None:
            progress.update(task_id, completed=i, c_id=f"{c_id} [{status}]")

        metrics = run_evaluation(
            orchestrator=orchestrator,
            contracts=contracts,
            output_dir=output_dir,
            limit=args.limit,
            progress_cb=_cb,
            playbook_path=Path(args.playbook) if args.playbook else None,
            ms1_csv_path=Path(args.ms1_csv) if args.ms1_csv else None,
            shard_index=args.shard_index,
            shard_total=args.shard_total,
        )

    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.table import Table

    table = Table(
        show_header=False, box=ROUNDED, border_style=BRAND_MUTED, expand=False
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
    con.print(
        Panel(
            table,
            title=f"[{BRAND_PRIMARY}]Aggregate metrics[/]",
            title_align="left",
            border_style=BRAND_PRIMARY,
            box=ROUNDED,
            padding=(0, 1),
        )
    )

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


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="BetterCallNLI",
        description=(
            "Claude Code-style CLI for the BetterCallNLI NDA review agent.\n\n"
            "Run without --mode (or with --mode converse) to start an interactive session\n"
            "where contract and intent are set per-prompt via @ref syntax and IntentRouter.\n"
            "When hypothesis_analysis intent is detected, a picker lets you choose which\n"
            "hypotheses to run before the pipeline starts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["converse", "analyze", "evaluate"],
        default=None,
        help=(
            "Session mode. Omit to start an interactive session with per-prompt routing. "
            "'converse' locks intent to conversation (backwards compatible). "
            "'analyze' runs a one-shot 17-hypothesis review (requires --contract). "
            "'evaluate' runs batch evaluation on the ContractNLI test split."
        ),
    )
    p.add_argument(
        "--retrieval",
        choices=["vector", "graphrag"],
        default="graphrag",
        help="Retriever backend (default: graphrag).",
    )
    p.add_argument(
        "--contract",
        help=(
            "Path to a .txt contract OR a test-set contract ID. "
            "Required for --mode analyze. Optional for interactive/converse — "
            "can also be set per-prompt with @path/to/contract.txt or @id."
        ),
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
    p.add_argument("--output",     help="(analyze) write final result JSON here.")
    p.add_argument(
        "--output-dir",
        default="results/ms3",
        help="(evaluate) output directory.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(evaluate) cap number of contracts.",
    )
    p.add_argument(
        "--save-history",
        help="(interactive/converse) write ConversationHistory JSON on every turn.",
    )
    p.add_argument(
        "--session-runtrace",
        help="(interactive/converse) explicit path for the per-session runtrace (§2c).",
    )
    p.add_argument(
        "--ms1-csv",
        default="results/evaluation_metrics.csv",
        help="(evaluate) existing MS1 CSV to merge into the combined CSV (§5b).",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="(evaluate) 0-indexed shard for parallel runs (default 0).",
    )
    p.add_argument(
        "--shard-total",
        type=int,
        default=1,
        help="(evaluate) total number of shards (default 1).",
    )
    p.add_argument(
        "--show-cards",
        action="store_true",
        help="Print one verdict panel per hypothesis (interactive/analyze). Toggleable at runtime with 'cards on/off'.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print tool-call traces after each turn.",
    )
    return p


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()

    # analyze and evaluate are still non-interactive dedicated modes
    if args.mode == "analyze":
        return run_analyze(args)
    if args.mode == "evaluate":
        return run_evaluate(args)

    # Everything else (None or "converse") goes to the interactive session.
    # run_interactive reads args.mode to know whether intent is locked.
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())