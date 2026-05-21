"""
Rich-based terminal UI for the BetterCallNLI CLI.

Provides:
  - Console singleton with project-wide styling
  - Banner rendering (ASCII logo)
  - Conversation / hypothesis result cards
  - status() context manager — animated spinner + final ✓ line
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── colour palette ────────────────────────────────────────────────────────────

BRAND_PRIMARY  = "bold #ff8c42"
BRAND_ACCENT   = "#5cc8ff"
BRAND_MUTED    = "grey62"
OK_STYLE       = "bold green"
WARN_STYLE     = "bold yellow"
ERR_STYLE      = "bold red"

LABEL_COLOURS = {
    "ENTAILED":      "bold green",
    "CONTRADICTED":  "bold red",
    "NOT_MENTIONED": "bold yellow",
}

MODE_COLOURS = {
    "conversation":        "bold cyan",
    "hypothesis_analysis": "bold magenta",
}


# ── singleton console ─────────────────────────────────────────────────────────

class Console(RichConsole):
    """rich.Console preconfigured for the BetterCallNLI brand."""

    def __init__(self) -> None:
        super().__init__(highlight=False, soft_wrap=False)


_console: Optional[Console] = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console()
    return _console


# ── banner ────────────────────────────────────────────────────────────────────

_BANNER_LINES: List[str] = [
    "  ____       _   _            ____      _ _ _   _ _     ___ ",
    " | __ )  ___| |_| |_ ___ _ __/ ___|__ _| | | \\ | | |   |_ _|",
    " |  _ \\ / _ \\ __| __/ _ \\ '__| |   / _` | | |  \\| | |    | | ",
    " | |_) |  __/ |_| ||  __/ |  | |__| (_| | | | |\\  | |___ | | ",
    " |____/ \\___|\\__|\\__\\___|_|   \\____\\__,_|_|_|_| \\_|_____|___|",
]


def render_banner(version: str = "0.3.0", subtitle: str = "NDA Review Agent") -> None:
    """Print the ASCII banner with version + subtitle, Claude Code style."""
    con = get_console()

    logo = Text("\n".join(_BANNER_LINES), style=BRAND_PRIMARY)
    sub  = Text(f"v{version}  ·  {subtitle}", style=BRAND_ACCENT)
    tip  = Text(
        "Type your question, or 'analyze this contract' for the full 17-hypothesis review.\n"
        "Type 'exit' or press Ctrl+C to quit.",
        style=BRAND_MUTED,
    )

    body = Text()
    body.append(logo)
    body.append("\n\n")
    body.append(sub)
    body.append("\n\n")
    body.append(tip)

    con.print(Panel(body, border_style=BRAND_PRIMARY, box=ROUNDED, padding=(1, 2)))


# ── status spinner ────────────────────────────────────────────────────────────

@contextmanager
def status(message: str, *, ok_message: Optional[str] = None):
    """
    Animated spinner that prints a final ✓ line when the with-block exits cleanly,
    and ✗ on failure.

        with status("Routing intent…", ok_message="Intent classified"):
            intent = router.route(msg)
    """
    con = get_console()
    with con.status(f"[{BRAND_ACCENT}]{message}", spinner="dots") as st:
        try:
            yield st
        except Exception:
            con.print(f"[{ERR_STYLE}]✗[/] {message}")
            raise
    con.print(f"[{OK_STYLE}]✓[/] {ok_message or message}")


# ── result renderers ──────────────────────────────────────────────────────────

def render_conversation_result(result: Dict[str, Any]) -> None:
    """Render a ConversationAgent result card (response + evidence + precedents)."""
    con = get_console()

    response: str            = result.get("response", "")
    evidence: List[Dict]     = result.get("evidence", []) or []
    precedents: List[Dict]   = result.get("precedents", []) or []
    retrieval_mode: str      = result.get("retrieval_mode", "?")
    usage: Dict              = result.get("usage", {}) or {}

    con.print()
    con.print(Panel(
        Text(response),
        title=f"[{BRAND_PRIMARY}]Agent[/] · [{MODE_COLOURS['conversation']}]conversation[/]",
        title_align="left",
        border_style=MODE_COLOURS["conversation"],
        box=ROUNDED,
        padding=(1, 2),
    ))

    if evidence:
        ev_table = Table(
            show_header=True, header_style=BRAND_ACCENT,
            box=ROUNDED, border_style=BRAND_MUTED, expand=False,
        )
        ev_table.add_column("#", justify="right", style=BRAND_MUTED, width=3)
        ev_table.add_column("✓", justify="center", width=2)
        ev_table.add_column("Quote (contract)", overflow="fold")
        ev_table.add_column("Pos", style=BRAND_MUTED, justify="right")

        for i, ev in enumerate(evidence, 1):
            verified = ev.get("verified", False)
            icon     = "[green]✓[/]" if verified else "[yellow]![/]"
            quote    = ev.get("quote", "")
            quote    = (quote[:120] + "…") if len(quote) > 120 else quote
            pos      = f"{ev.get('char_start', -1)}–{ev.get('char_end', -1)}" if ev.get("char_start", -1) >= 0 else "—"
            ev_table.add_row(str(i), icon, quote, pos)

        con.print()
        con.print(Panel(
            ev_table,
            title=f"[{BRAND_ACCENT}]Evidence ({len(evidence)})[/]",
            title_align="left",
            border_style=BRAND_MUTED,
            box=ROUNDED,
            padding=(0, 1),
        ))

    if precedents:
        prec_table = Table(
            show_header=True, header_style=BRAND_ACCENT,
            box=ROUNDED, border_style=BRAND_MUTED, expand=False,
        )
        prec_table.add_column("#", justify="right", style=BRAND_MUTED, width=3)
        prec_table.add_column("Label", width=14)
        prec_table.add_column("Score", justify="right", width=6)
        prec_table.add_column("Snippet", overflow="fold")

        for i, p in enumerate(precedents, 1):
            label = p.get("label", "?")
            style = LABEL_COLOURS.get(label.upper(), BRAND_MUTED)
            score = f"{p.get('score', 0):.2f}"
            snippet = p.get("text", "")
            snippet = (snippet[:140] + "…") if len(snippet) > 140 else snippet
            prec_table.add_row(str(i), Text(label, style=style), score, snippet)

        con.print()
        con.print(Panel(
            prec_table,
            title=f"[{BRAND_ACCENT}]Retrieved precedents · {retrieval_mode}[/]",
            title_align="left",
            border_style=BRAND_MUTED,
            box=ROUNDED,
            padding=(0, 1),
        ))

    if usage:
        con.print(
            f"[{BRAND_MUTED}]usage · prompt {usage.get('prompt_tokens', 0)} tok"
            f" · completion {usage.get('completion_tokens', 0)} tok[/]"
        )


def render_verdict_card(verdict: Dict[str, Any], *, hypothesis_title: str = "") -> None:
    """Render one hypothesis verdict as a labelled panel."""
    con = get_console()

    label = (verdict.get("label") or "").upper()
    label_style = LABEL_COLOURS.get(label, BRAND_MUTED)
    confidence = float(verdict.get("confidence", 0.0))
    reasoning = verdict.get("reasoning", "") or "—"
    evidence  = verdict.get("evidence", []) or []
    h_id      = verdict.get("hypothesis_id", "?")

    header = Text()
    header.append(f"{h_id}", style="bold")
    if hypothesis_title:
        header.append(f"  ·  {hypothesis_title}", style=BRAND_MUTED)

    body = Text()
    body.append("label       ", style=BRAND_MUTED)
    body.append(label or "—", style=label_style)
    body.append("\n")
    body.append("confidence  ", style=BRAND_MUTED)
    body.append(f"{confidence:.2f}", style=BRAND_ACCENT)
    body.append("\n")
    body.append("evidence    ", style=BRAND_MUTED)
    body.append(f"{len(evidence)} span(s)", style=BRAND_ACCENT)
    body.append("\n\n")
    body.append("reasoning\n", style=BRAND_MUTED)
    body.append(reasoning.strip())

    for i, ev in enumerate(evidence, 1):
        body.append("\n\n")
        body.append(f"  [{i}] ", style=BRAND_MUTED)
        quote = ev.get("quote", "")
        quote = (quote[:240] + "…") if len(quote) > 240 else quote
        body.append(quote)

    con.print(Panel(
        body,
        title=header,
        title_align="left",
        border_style=label_style,
        box=ROUNDED,
        padding=(1, 2),
    ))


def render_hypothesis_summary(
    verdicts: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
) -> None:
    """Render a one-row-per-hypothesis summary table."""
    con = get_console()

    table = Table(
        show_header=True, header_style=BRAND_ACCENT,
        box=ROUNDED, border_style=BRAND_MUTED, expand=False,
    )
    table.add_column("Hyp", style="bold", width=4)
    table.add_column("Label", width=14)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Ev", justify="right", width=4)
    table.add_column("Reasoning", overflow="fold")

    for v in verdicts:
        label = (v.get("label") or "").upper()
        style = LABEL_COLOURS.get(label, BRAND_MUTED)
        reasoning = v.get("reasoning", "") or ""
        reasoning = (reasoning[:90] + "…") if len(reasoning) > 90 else reasoning

        table.add_row(
            v.get("hypothesis_id", "?"),
            Text(label or "—", style=style),
            f"{float(v.get('confidence', 0.0)):.2f}",
            str(len(v.get("evidence", []) or [])),
            reasoning,
        )

    con.print()
    con.print(Panel(
        table,
        title=f"[{BRAND_PRIMARY}]{title or 'Hypothesis verdicts'}[/]",
        title_align="left",
        border_style=BRAND_PRIMARY,
        box=ROUNDED,
        padding=(0, 1),
    ))


# ── tool-call trace renderer ──────────────────────────────────────────────────

def render_tool_calls(tool_calls: Iterable[Dict[str, Any]]) -> None:
    """Compact per-tool-call trace (debug aid for --verbose)."""
    con = get_console()
    table = Table(
        show_header=True, header_style=BRAND_ACCENT,
        box=ROUNDED, border_style=BRAND_MUTED, expand=False,
    )
    table.add_column("Tool", style="bold")
    table.add_column("ms", justify="right", width=8)
    table.add_column("Output summary", overflow="fold")

    for tc in tool_calls:
        name = tc.get("name", "?")
        latency = tc.get("latency_ms", 0.0)
        out = tc.get("output", {})
        if isinstance(out, dict):
            summary = " · ".join(f"{k}={v}" for k, v in list(out.items())[:4])
        else:
            summary = str(out)[:120]
        table.add_row(name, f"{latency:.0f}", summary)

    con.print(Panel(
        table,
        title=f"[{BRAND_ACCENT}]Tool trace ({sum(1 for _ in tool_calls)})[/]",
        title_align="left",
        border_style=BRAND_MUTED,
        box=ROUNDED,
        padding=(0, 1),
    ))
