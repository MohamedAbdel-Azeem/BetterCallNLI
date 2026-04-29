"""
GraphRAG retriever — wraps the existing Neo4j pipeline from graphrag_utils.py.

Two query modes:
  - hypothesis-scoped  (hypothesis_id given): graph-filter by hypothesis then vector-rank
  - free-form          (hypothesis_id=None):  pure vector search across all Clause nodes
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

# Make notebooks/scripts importable without changing the project's package layout
_SCRIPTS_DIR = str(Path(__file__).parent.parent.parent / "notebooks" / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from graphrag_utils import get_driver  # noqa: E402

_EMBED_MODEL = "all-MiniLM-L6-v2"
_MIN_SCORE = 0.45  # lower than classification (0.7) to cast a wider net for conversation

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

_HYPOTHESIS_QUERY = """
MATCH (a:Annotation)-[:GROUNDS_HYPOTHESIS]->(h:Hypothesis {h_id: $h_id})
MATCH (a)-[:SUPPORTED_BY]->(cl:Clause)
WITH cl, a.label AS label,
     vector.similarity.cosine(cl.embedding, $embedding) AS score
WHERE score >= $min_score
ORDER BY score DESC
LIMIT $limit
RETURN cl.clause_id AS clause_id,
       cl.text       AS text,
       label,
       score
"""


class GraphRAGRetriever(BaseRetriever):
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None
        self._embedder: Optional[SentenceTransformer] = None
        self._ready = False

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return "graphrag"

    def is_ready(self) -> bool:
        return self._ready

    def connect(self) -> bool:
        """Open Neo4j connection and load the embedding model. Returns success."""
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
    ) -> List[Dict[str, Any]]:
        if not self._ready:
            return []

        emb = self._get_embedder().encode([query])[0].tolist()

        with self._driver.session() as s:
            if hypothesis_id:
                rows = s.run(
                    _HYPOTHESIS_QUERY,
                    {
                        "embedding": emb,
                        "h_id": hypothesis_id,
                        "min_score": _MIN_SCORE,
                        "limit": k,
                    },
                ).data()
            else:
                rows = s.run(
                    _FREEFORM_QUERY,
                    {"embedding": emb, "min_score": _MIN_SCORE, "limit": k},
                ).data()

        return [
            {
                "text": r["text"],
                "clause_id": r.get("clause_id"),
                "label": r.get("label", "UNKNOWN"),
                "score": round(float(r["score"]), 4),
                "hypothesis_id": hypothesis_id or "—",
            }
            for r in rows
        ]

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(_EMBED_MODEL)
        return self._embedder
