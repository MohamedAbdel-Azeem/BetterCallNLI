"""
GraphRAG retriever — hybrid pipeline.

High-confidence hypothesis match (>= 0.75):
    auto-detect nearest hypothesis → get_anchors (contract chunks vs hypothesis embedding)
    → get_precedents (ENTAILED + CONTRADICTED from graph)

Everything else:
    free-form Neo4j vector search fallback
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

_SCRIPTS_DIR = str(Path(__file__).parent.parent.parent /  "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from graphrag_utils import (  # noqa: E402
    get_driver,
    get_eval_chunks,
    get_anchors,
    get_precedents,
    fetch_hypothesis,
)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_MIN_SCORE = 0.7
_HYPOTHESIS_THRESHOLD = 0.80  # only route to full pipeline when clearly hypothesis-shaped

_ALL_HYPOTHESES_QUERY = """
MATCH (h:Hypothesis)
RETURN h.h_id AS h_id, h.embedding AS embedding
"""

_FREEFORM_QUERY = """
MATCH (a:Annotation)-[:SUPPORTED_BY]->(cl:Clause)
WITH cl, a.label AS label,
     vector.similarity.cosine(cl.embedding, $embedding) AS score
WHERE score >= $min_score
WITH cl, label, score
ORDER BY score DESC
WITH cl.clause_id AS clause_id, cl.text AS text,
     head(collect(label)) AS label,
     head(collect(score)) AS score
ORDER BY score DESC
LIMIT $limit
RETURN clause_id, text, label, score
"""


class GraphRAGRetriever(BaseRetriever):
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None
        self._embedder: Optional[SentenceTransformer] = None
        self._ready = False
        self._hypothesis_cache: Optional[Dict[str, np.ndarray]] = None

    @property
    def mode(self) -> str:
        return "graphrag"

    def is_ready(self) -> bool:
        return self._ready

    def connect(self) -> bool:
        try:
            self._driver = get_driver(self.uri, self.username, self.password)
            self._embedder = SentenceTransformer(_EMBED_MODEL)
            self._ready = True
            return True
        except Exception as exc:
            print(f"[GraphRAGRetriever] connection failed: {exc}")
            self._ready = False
            return False

    def retrieve(
        self,
        query: str,
        hypothesis_id: Optional[str] = None,
        k: int = 5,
        contract: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._ready:
            return []

        query_emb = self._get_embedder().encode([query])[0]

        if hypothesis_id:
            h_id, confidence = hypothesis_id, 1.0
        else:
            h_id, confidence = self._find_nearest_hypothesis(query_emb)

        if confidence >= _HYPOTHESIS_THRESHOLD and contract:
            return self._pipeline_retrieve(contract, h_id, k)

        return self._freeform_retrieve(query_emb.tolist(), k)

    # ── private helpers ───────────────────────────────────────────────────────

    def _find_nearest_hypothesis(self, query_emb: np.ndarray) -> Tuple[str, float]:
        hyps = self._load_hypothesis_embeddings()
        if not hyps:
            return "", 0.0
        best_h_id, best_score = "", -1.0
        for h_id, h_emb in hyps.items():
            score = float(
                np.dot(query_emb, h_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(h_emb) + 1e-9)
            )
            if score > best_score:
                best_score = score
                best_h_id = h_id
        return best_h_id, best_score

    def _load_hypothesis_embeddings(self) -> Dict[str, np.ndarray]:
        if self._hypothesis_cache is not None:
            return self._hypothesis_cache
        try:
            with self._driver.session() as s:
                rows = s.run(_ALL_HYPOTHESES_QUERY).data()
            self._hypothesis_cache = {
                r["h_id"]: np.array(r["embedding"]) for r in rows
            }
        except Exception as exc:
            print(f"[GraphRAGRetriever] failed to load hypothesis embeddings: {exc}")
            self._hypothesis_cache = {}
        return self._hypothesis_cache

    def _pipeline_retrieve(
        self, contract: Dict[str, Any], h_id: str, k: int
    ) -> List[Dict[str, Any]]:
        h_data = fetch_hypothesis(h_id, self._driver)
        chunks = get_eval_chunks(contract)
        anchors = get_anchors(
            chunks, h_data["embedding"], self._get_embedder(), top_k=3
        )
        precedents = get_precedents(
            anchors, h_id, self._driver, per_label=max(k // 2, 2)
        )

        results: List[Dict[str, Any]] = []
        for label, items in precedents.items():
            for p in items:
                results.append({
                    "text": p["text"],
                    "clause_id": p.get("clause_id"),
                    "label": label,
                    "score": round(float(p["score"]), 4),
                    "hypothesis_id": h_id,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _freeform_retrieve(self, emb_list: list, k: int) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            rows = s.run(
                _FREEFORM_QUERY,
                {"embedding": emb_list, "min_score": _MIN_SCORE, "limit": k},
            ).data()
        return [
            {
                "text": r["text"],
                "clause_id": r.get("clause_id"),
                "label": r.get("label", "UNKNOWN"),
                "score": round(float(r["score"]), 4),
                "hypothesis_id": "—",
            }
            for r in rows
        ]

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(_EMBED_MODEL)
        return self._embedder
