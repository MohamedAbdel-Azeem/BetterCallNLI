"""
Vector RAG retriever — backed by ChromaDB Cloud.

Connects to the 'nda_chunks' collection populated by the Vector RAG notebook
(scripts/vector_rag_query.py).  Auto-connects on init; is_ready() returns
False if CHROMA_API_KEY is missing or the connection fails.

Retrieval logic mirrors _query_chroma_for_examples() from vector_rag_query.py:
  - embed the query string
  - fetch candidates from ChromaDB Cloud
  - filter by hypothesis_id (nda-* format) when requested
  - return results with similarity scores
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
import chromadb

from .base import BaseRetriever

# Make scripts/ importable so callers can use build_rag_prompt directly if needed
_SCRIPTS_DIR = str(Path(__file__).parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# ── ChromaDB Cloud config (matches vector_rag_setup.ipynb) ───────────────────
_CHROMA_TENANT   = "fd2eb954-04b9-4e06-adbe-b687d3c0d25b"
_CHROMA_DATABASE = "Agentic_project"
_COLLECTION_NAME = "nda_chunks"
_EMBED_MODEL     = "all-MiniLM-L6-v2"

# Distance ceiling for ChromaDB cosine results (mirrors chroma_distance_threshold=0.3
# used in vector_rag_query.py; slightly relaxed here for free-form conversation)
_DIST_THRESHOLD = 0.45
_MAX_FETCH      = 20   # over-fetch then filter down to k

# ── H01-H17 (playbook IDs) → nda-* (ContractNLI / ChromaDB metadata IDs) ─────
# nda-6, nda-9, nda-14 are excluded from ContractNLI's 17-hypothesis subset
_H_TO_NDA: Dict[str, str] = {
    "H01": "nda-1",  "H02": "nda-2",  "H03": "nda-3",
    "H04": "nda-4",  "H05": "nda-5",  "H06": "nda-7",
    "H07": "nda-8",  "H08": "nda-10", "H09": "nda-11",
    "H10": "nda-12", "H11": "nda-13", "H12": "nda-15",
    "H13": "nda-16", "H14": "nda-17", "H15": "nda-18",
    "H16": "nda-19", "H17": "nda-20",
}


class VectorRAGRetriever(BaseRetriever):
    """
    Retrieves NDA precedents from the ChromaDB Cloud collection built by
    the Vector RAG notebook (scripts/vector_rag_query.py).
    """

    def __init__(self) -> None:
        self._collection = None
        self._embedder: Optional[SentenceTransformer] = None
        self._ready = False
        self._try_connect()

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return "vector"

    def is_ready(self) -> bool:
        return self._ready

    def collection_count(self) -> int:
        return self._collection.count() if self._collection else 0

    def retrieve(
        self,
        query: str,
        hypothesis_id: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant NDA precedents for a free-form query string.

        Args:
            query:         user question or contract excerpt to match against
            hypothesis_id: optional H01-H17 filter; when given only precedents
                           for that hypothesis are returned
            k:             max results to return

        Returns:
            list of dicts with keys: text, label, score, hypothesis_id
        """
        if not self._ready or self._collection is None:
            return []

        query_emb = self._embedder.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_emb,
            n_results=_MAX_FETCH,
            include=["documents", "metadatas", "distances"],
        )

        # Convert playbook H-id to nda-* format used in ChromaDB metadata
        nda_id = _H_TO_NDA.get(hypothesis_id) if hypothesis_id else None

        output: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if text in seen or dist > _DIST_THRESHOLD:
                continue

            evidence_for: List[Dict] = json.loads(meta.get("evidence_for", "[]"))

            if nda_id:
                # Hypothesis-scoped: only keep if this hypothesis is in evidence_for
                matched = next(
                    (e for e in evidence_for if e.get("hypothesis_id") == nda_id),
                    None,
                )
                if not matched:
                    continue
                label = matched.get("gold_label", "UNKNOWN")
            else:
                # Free-form: take the label from the first entry
                label = evidence_for[0].get("gold_label", "UNKNOWN") if evidence_for else "UNKNOWN"

            seen.add(text)
            output.append(
                {
                    "text": text,
                    "label": label,
                    "score": round(1.0 - float(dist), 4),
                    "hypothesis_id": hypothesis_id or "—",
                }
            )

            if len(output) >= k:
                break

        return output

    # ── private ───────────────────────────────────────────────────────────────

    def _try_connect(self) -> None:
        api_key = os.getenv("CHROMA_API_KEY", "").strip("'\"")
        if not api_key:
            return
        try:
            client = chromadb.CloudClient(
                api_key=api_key,
                tenant=_CHROMA_TENANT,
                database=_CHROMA_DATABASE,
            )
            self._collection = client.get_collection(_COLLECTION_NAME)
            self._embedder = SentenceTransformer(_EMBED_MODEL)
            self._ready = True
        except Exception as exc:
            print(f"[VectorRAGRetriever] ChromaDB Cloud connection failed: {exc}")
