"""
BetterCallNLI — Streamlit Conversation Agent UI
================================================
Run:  streamlit run app.py
Env:  HF_TOKEN, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, CHROMA_API_KEY  (in .env)

Layout
------
Sidebar  : contract loader, retrieval-mode switch, session controls, status panel
Left col : contract viewer with colour-coded evidence highlights
Right col: chat interface (messages + sticky input)
"""

from __future__ import annotations

import html
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.conversation_agent import ConversationAgent
from src.agent.history import ConversationHistory
from src.retrieval.graphrag_retriever import GraphRAGRetriever
from src.retrieval.vector_rag import VectorRAGRetriever
from src.utils.contract_loader import (
    contract_from_text,
    get_test_contracts,
)

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BetterCallNLI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .contract-viewer {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-y: auto;
        max-height: 72vh;
        padding: 10px 14px;
        background: #000000;
        border: 1px solid #dee2e6;
        border-radius: 6px;
    }
    .badge-vector  { background:#0d6efd; color:white; padding:2px 8px; border-radius:10px; font-size:12px; }
    .badge-graphrag{ background:#198754; color:white; padding:2px 8px; border-radius:10px; font-size:12px; }
    .evidence-card { padding:6px 10px; border-radius:4px; margin:3px 0; font-size:12px; }
    .ev-ok  { background:#d1e7dd; }
    .ev-warn{ background:#fff3cd; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── highlight colours (one per assistant turn, cycles) ────────────────────────

_HIGHLIGHT_COLORS = [
    "#FFD700", "#98FB98", "#87CEEB", "#DDA0DD",
    "#F08080", "#FFDAB9", "#B0E0E6", "#E6E6FA",
]


# ── cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Neo4j GraphRAG…")
def _build_graphrag_retriever() -> Optional[GraphRAGRetriever]:
    uri  = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "")
    pwd  = os.getenv("NEO4J_PASSWORD", "")
    if not all([uri, user, pwd]):
        return None
    r = GraphRAGRetriever(uri=uri, username=user, password=pwd)
    r.connect()
    return r


@st.cache_resource(show_spinner="Initialising Vector RAG…")
def _build_vector_retriever() -> VectorRAGRetriever:
    return VectorRAGRetriever()


# ── helpers ───────────────────────────────────────────────────────────────────

def _render_contract(text: str, highlights: List[Tuple[int, int, int]]) -> str:
    """Return HTML string of contract text with colour-coded evidence spans."""
    if not highlights:
        return f'<div class="contract-viewer">{html.escape(text)}</div>'

    events: List[Tuple[int, str, str]] = []
    for start, end, color_idx in highlights:
        color = _HIGHLIGHT_COLORS[color_idx % len(_HIGHLIGHT_COLORS)]
        events.append((start, "open",  color))
        events.append((end,   "close", color))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "close" else 1))

    parts: List[str] = []
    pos = 0
    for ev_pos, ev_type, color in events:
        if pos < ev_pos:
            parts.append(html.escape(text[pos:ev_pos]))
        if ev_type == "open":
            parts.append(
                f'<mark style="background:{color};padding:0 2px;border-radius:2px;">'
            )
        else:
            parts.append("</mark>")
        pos = max(pos, ev_pos)
    if pos < len(text):
        parts.append(html.escape(text[pos:]))

    return f'<div class="contract-viewer">{"".join(parts)}</div>'


def _collect_highlights(messages: List[Dict]) -> List[Tuple[int, int, int]]:
    """Aggregate (start, end, color_idx) from all assistant turns."""
    out = []
    color_idx = 0
    for msg in messages:
        if msg["role"] == "assistant":
            for ev in msg.get("evidence", []):
                if ev.get("verified") and ev.get("char_start", -1) >= 0:
                    out.append((ev["char_start"], ev["char_end"], color_idx))
            color_idx += 1
    return out


def _reset_session() -> None:
    st.session_state.messages  = []
    st.session_state.history   = ConversationHistory()
    st.session_state.session_id = str(uuid.uuid4())[:8]


def _set_contract(contract: Dict[str, Any]) -> None:
    st.session_state.contract = contract
    _reset_session()


# ── session-state bootstrap ───────────────────────────────────────────────────

if "messages"       not in st.session_state: st.session_state.messages    = []
if "contract"       not in st.session_state: st.session_state.contract    = None
if "history"        not in st.session_state: st.session_state.history     = ConversationHistory()
if "session_id"     not in st.session_state: st.session_state.session_id  = str(uuid.uuid4())[:8]
if "retrieval_mode" not in st.session_state: st.session_state.retrieval_mode = "graphrag"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚖️ BetterCallNLI")
    st.caption(f"Session `{st.session_state.session_id}`")
    st.divider()

    # ── Contract source ───────────────────────────────────────────────────────
    st.subheader("📄 Contract")
    source = st.radio(
        "source",
        ["Upload .txt", "Paste text", "Test-set contract"],
        label_visibility="collapsed",
    )

    if source == "Upload .txt":
        f = st.file_uploader("NDA file", type=["txt"], label_visibility="collapsed")
        if f:
            text = f.read().decode("utf-8", errors="ignore")
            c = contract_from_text(text, contract_id=Path(f.name).stem)
            if st.session_state.contract != c:
                _set_contract(c)
                st.success(f"Loaded **{f.name}** — {len(text):,} chars")

    elif source == "Paste text":
        pasted = st.text_area(
            "Contract text", height=180,
            placeholder="Paste your NDA here…",
            label_visibility="collapsed",
        )
        if st.button("Load", use_container_width=True) and pasted.strip():
            _set_contract(contract_from_text(pasted))
            st.success(f"Loaded — {len(pasted):,} chars")

    else:  # Test-set contract
        local_path = st.text_input(
            "Dataset directory (blank = kagglehub)",
            placeholder="/path/to/contractnli/",
        )
        if st.button("Fetch test contracts", use_container_width=True):
            with st.spinner("Loading test set…"):
                try:
                    contracts = get_test_contracts(local_path=local_path or None)
                    st.session_state.test_contracts = contracts
                    st.success(f"{len(contracts)} contracts loaded")
                except Exception as exc:
                    st.error(str(exc))

        if "test_contracts" in st.session_state:
            ids = [c["id"] for c in st.session_state.test_contracts]
            sel = st.selectbox("Contract ID", ids, label_visibility="collapsed")
            if st.button("Select contract", use_container_width=True):
                chosen = next(c for c in st.session_state.test_contracts if c["id"] == sel)
                _set_contract(chosen)

    st.divider()

    # ── Retrieval mode ────────────────────────────────────────────────────────
    st.subheader("🔍 Retrieval mode")
    mode_choice = st.radio(
        "mode",
        ["GraphRAG", "Vector RAG"],
        index=0 if st.session_state.retrieval_mode == "graphrag" else 1,
        label_visibility="collapsed",
    )
    st.session_state.retrieval_mode = "graphrag" if mode_choice == "GraphRAG" else "vector"

    if st.session_state.retrieval_mode == "graphrag":
        st.caption("🕸️ Neo4j graph-filter + cosine ranking on training clauses")
    else:
        st.caption("📦 ChromaDB Cloud semantic search on 32k+ training clause vectors")

    st.divider()

    # ── Session controls ──────────────────────────────────────────────────────
    st.subheader("💬 Session")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("New session", use_container_width=True):
            _reset_session()
            st.rerun()
    with c2:
        if st.session_state.history.turns:
            hist_json = json.dumps(
                st.session_state.history.to_dict(), indent=2, ensure_ascii=False
            )
            st.download_button(
                "Download history",
                data=hist_json,
                file_name=f"history_{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )
    st.caption(f"Turns in session: **{len(st.session_state.history)}**")

    st.divider()

    # ── Status panel ──────────────────────────────────────────────────────────
    st.subheader("🔑 Status")
    hf_ok     = bool(os.getenv("HF_TOKEN"))
    neo4j_ok  = bool(os.getenv("NEO4J_URI"))
    chroma_ok = bool(os.getenv("CHROMA_API_KEY"))

    st.write("HuggingFace API",       "✅" if hf_ok     else "❌ set HF_TOKEN in .env")
    st.write("Neo4j (GraphRAG)",      "✅" if neo4j_ok  else "❌ set NEO4J_* in .env")
    st.write("ChromaDB (Vector RAG)", "✅" if chroma_ok else "❌ set CHROMA_API_KEY in .env")

    # Live connection test — optional, runs on demand
    if st.button("🔌 Test retriever connections", use_container_width=True):
        col_g, col_v = st.columns(2)
        with col_g:
            with st.spinner("GraphRAG…"):
                try:
                    gr = _build_graphrag_retriever()
                    if gr and gr.is_ready():
                        st.success("GraphRAG ✅")
                    else:
                        st.error("GraphRAG ❌")
                except Exception as exc:
                    st.error(f"GraphRAG ❌\n{exc}")
        with col_v:
            with st.spinner("Vector RAG…"):
                try:
                    vr = _build_vector_retriever()
                    if vr.is_ready():
                        st.success(f"Vector RAG ✅")
                    else:
                        st.error("Vector RAG ❌")
                except Exception as exc:
                    st.error(f"Vector RAG ❌\n{exc}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — guard rails
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚖️ BetterCallNLI — NDA Conversation Agent")

if not os.getenv("HF_TOKEN"):
    st.error("**HF_TOKEN** is not set. Add your HuggingFace token to the `.env` file and restart.")
    st.stop()

if st.session_state.contract is None:
    st.info("👈 Load a contract from the sidebar to start a conversation.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([5, 6], gap="large")

# ── Left: contract viewer ─────────────────────────────────────────────────────

with left:
    contract = st.session_state.contract
    mode_tag = (
        '<span class="badge-graphrag">GraphRAG</span>'
        if st.session_state.retrieval_mode == "graphrag"
        else '<span class="badge-vector">Vector RAG</span>'
    )
    st.markdown(
        f"**Contract** `{contract['id']}` &nbsp;·&nbsp; "
        f"{contract.get('char_count', len(contract['text'])):,} chars &nbsp; {mode_tag}",
        unsafe_allow_html=True,
    )

    highlights = _collect_highlights(st.session_state.messages)
    st.markdown(
        _render_contract(contract["text"], highlights),
        unsafe_allow_html=True,
    )
    if highlights:
        st.caption(f"🔆 {len(highlights)} evidence span(s) highlighted from conversation")

# ── Right: chat interface ──────────────────────────────────────────────────────

with right:
    st.subheader("💬 Conversation")

    chat_box = st.container(height=560)
    with chat_box:
        if not st.session_state.messages:
            st.info(
                "Ask anything about this contract. Examples:\n"
                "- *What are the main confidentiality obligations?*\n"
                "- *Can the receiving party share information with employees?*\n"
                "- *What happens to confidential information after termination?*\n"
                "- *Is there a non-solicitation clause?*"
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

                if msg["role"] == "assistant":
                    evidence  = msg.get("evidence",  [])
                    precedents = msg.get("precedents", [])
                    r_mode    = msg.get("retrieval_mode", "?")

                    if evidence:
                        verified   = [e for e in evidence if e.get("verified")]
                        unverified = [e for e in evidence if not e.get("verified")]
                        label = (
                            f"📎 {len(verified)} verified"
                            + (f", {len(unverified)} unverified" if unverified else "")
                            + " evidence citation(s)"
                        )
                        with st.expander(label):
                            for i, ev in enumerate(evidence, 1):
                                icon = "✅" if ev.get("verified") else "⚠️"
                                note = f" *({ev['note']})*" if ev.get("note") else ""
                                st.markdown(
                                    f"{icon} **Evidence {i}**{note}  \n"
                                    f"> {ev['quote'][:300]}"
                                    + ("…" if len(ev["quote"]) > 300 else "")
                                )
                                if ev.get("char_start", -1) >= 0:
                                    st.caption(
                                        f"Position {ev['char_start']}–{ev['char_end']}"
                                    )

                    if precedents:
                        with st.expander(
                            f"🔍 {len(precedents)} retrieved precedent(s) via {r_mode}"
                        ):
                            for p in precedents:
                                st.markdown(
                                    f"**{p.get('label','?')}** · score {p.get('score',0):.2f}  \n"
                                    f"{p['text'][:250]}"
                                    + ("…" if len(p["text"]) > 250 else "")
                                )
                                st.divider()

# ── Chat input (sticky at page bottom) ───────────────────────────────────────

if prompt := st.chat_input("Ask about this NDA contract…"):
    # Validate retriever availability before doing anything
    if st.session_state.retrieval_mode == "vector":
        retriever = _build_vector_retriever()
        if not retriever.is_ready():
            st.warning(
                "Vector RAG could not connect to ChromaDB Cloud. "
                "Check `CHROMA_API_KEY` in your `.env`."
            )
            st.stop()
    else:
        retriever = _build_graphrag_retriever()
        if retriever is None or not retriever.is_ready():
            st.warning(
                "GraphRAG could not connect. Check `NEO4J_*` variables in your `.env`."
            )
            st.stop()

    # Append user message optimistically
    st.session_state.messages.append({"role": "user", "content": prompt})

    agent = ConversationAgent(
        retriever=retriever,
        hf_token=os.environ["HF_TOKEN"],
    )

    with st.spinner("Analysing contract…"):
        try:
            result = agent.chat(
                contract=st.session_state.contract,
                user_prompt=prompt,
                history=st.session_state.history,
            )
        except Exception as exc:
            st.session_state.messages.pop()  # undo optimistic append
            st.error(f"Agent error: {exc}")
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "evidence": result["evidence"],
            "precedents": result["precedents"],
            "retrieval_mode": result["retrieval_mode"],
        }
    )
    st.rerun()
