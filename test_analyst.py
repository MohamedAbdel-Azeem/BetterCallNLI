"""
Quick test: run HypothesisAnalyst on one contract + one hypothesis.
Usage:  python test_analyst.py
"""

import json
import os
import sys

# force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_rag import VectorRAGRetriever
from src.agent.hypothesis_analyst import HypothesisAnalyst, _extract_json
from src.utils.contract_loader import get_test_contracts

# ── config ────────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL      = "Qwen/Qwen2.5-7B-Instruct"   # confirmed available on HF Serverless
HYPOTHESIS = {
    "id":    "H04",
    "title": "Purpose limitation",
    "text":  "Receiving Party shall not use any Confidential Information for any purpose "
             "other than the purposes stated in Agreement.",
}

# ── retriever ─────────────────────────────────────────────────────────────────
print("Connecting to ChromaDB...")
retriever = VectorRAGRetriever()
if not retriever.is_ready():
    raise RuntimeError("VectorRAGRetriever not ready -- check CHROMA_API_KEY")
print(f"  collection size: {retriever.collection_count()} chunks\n")

# ── contract ──────────────────────────────────────────────────────────────────
print("Loading first test contract...")
contracts = get_test_contracts()
contract  = contracts[0]
print(f"  contract id : {contract['id']}")
print(f"  length      : {contract['char_count']} chars")
print(f"  preview     : {contract['text'][:200].strip()!r}\n")

# ── analyst ───────────────────────────────────────────────────────────────────
analyst = HypothesisAnalyst(retriever=retriever, hf_token=HF_TOKEN, model=MODEL)

print(f"Running analyst on {HYPOTHESIS['id']}: {HYPOTHESIS['title']}")
print("-" * 60)

verdict, tool_calls = analyst.analyze(
    contract=contract,
    hypothesis=HYPOTHESIS,
    attempt=1,
)

# ── results ───────────────────────────────────────────────────────────────────
print(f"Label           : {verdict['label']}")
print(f"Confidence      : {verdict['confidence']:.2f}")
print(f"Reasoning       : {verdict['reasoning']}")
print(f"Evidence spans  : {len(verdict['evidence'])}")
for i, e in enumerate(verdict['evidence'], 1):
    score = e.get('relevance_score', 0)
    print(f"  [{i}] score={score:.2f}  chars {e['char_start']}-{e['char_end']}")
    print(f"       {e['quote'][:120]!r}")
print(f"Counter evidence: {len(verdict['counter_evidence'])} quote(s)")
for i, c in enumerate(verdict['counter_evidence'], 1):
    print(f"  [{i}] {c[:120]!r}")

print("\n-- Tool calls --")
for tc in tool_calls:
    print(json.dumps(tc, indent=2, ensure_ascii=False))
