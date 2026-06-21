"""
GraphRAG pipeline utilities for BetterCallNLI.

Usage (peer entry point):
    from graphrag_utils import get_eval_chunks, get_anchors, get_precedents, build_prompt
"""

import numpy as np
from numpy.linalg import norm
from neo4j import GraphDatabase


# ── Neo4j setup ───────────────────────────────────────────────────────────────

def get_driver(uri: str, user: str, password: str):
    """
    Create and verify a Neo4j driver connection.

    Args:
        uri:      e.g. "neo4j+s://xxxx.databases.neo4j.io"
        user:     Neo4j username
        password: Neo4j password

    Returns:
        neo4j.Driver — verified live connection
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as s:
        result = s.run("RETURN 1 AS ping").single()
        assert result["ping"] == 1, "Neo4j connection failed"
    print(f"Neo4j connected: {uri}")
    return driver


# ── Default retrieval limits (override by passing kwargs) ─────────────────────
CHUNK_WINDOW  = 70
CHUNK_STRIDE  = 35
ANCHOR_TOP_K  = 3
PER_LABEL_K   = 5
MAX_FEW_SHOTS = 5

# ── Cypher query (graph-filter first, then cosine rank within filtered set) ───
# Graph filter runs first → only clauses annotated for this hypothesis + label.
# vector.similarity.cosine() then ranks that small set — nothing is missed.
_PRECEDENT_QUERY = """
MATCH (a:Annotation)-[:GROUNDS_HYPOTHESIS]->(h:Hypothesis {h_id: $h_id})
WHERE a.label = $label
MATCH (a)-[:SUPPORTED_BY]->(cl:Clause)
WITH cl, vector.similarity.cosine(cl.embedding, $embedding) AS score
WHERE score >= $min_score
ORDER BY score DESC
LIMIT $per_label
RETURN cl.clause_id AS clause_id,
       cl.text      AS text,
       score
"""

MIN_SCORE = 0.7  # default similarity threshold — lower = more results, higher = stricter


# ── Section 4 ─────────────────────────────────────────────────────────────────

def get_eval_chunks(contract: dict, window: int = CHUNK_WINDOW,
                    stride: int = CHUNK_STRIDE) -> list:
    """
    Chunk an eval contract for retrieval.
    Uses contract["spans"] ([start, end] pairs) if present;
    falls back to overlapping word-window sliding.
    Returns list of {chunk_id, text, char_start, char_end}.
    """
    text  = contract["text"]
    spans = contract.get("spans", [])

    if spans:
        return [
            {
                "chunk_id":   f"span_{idx}",
                "text":       text[span[0]:span[1]],
                "char_start": span[0],
                "char_end":   span[1],
            }
            for idx, span in enumerate(spans)
        ]

    words, chunks, start_w, chunk_idx = text.split(), [], 0, 0
    while start_w < len(words):
        end_w       = min(start_w + window, len(words))
        chunk_words = words[start_w:end_w]
        chunk_text  = " ".join(chunk_words)
        char_start  = text.find(chunk_words[0]) if chunk_words else 0
        chunks.append({
            "chunk_id":   f"window_{chunk_idx}",
            "text":       chunk_text,
            "char_start": char_start,
            "char_end":   char_start + len(chunk_text),
        })
        start_w += stride; chunk_idx += 1
        if end_w == len(words):
            break
    return chunks


# ── Section 5 ─────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))


def get_anchors(eval_chunks: list, hypothesis_embedding: np.ndarray,
                embedder, top_k: int = ANCHOR_TOP_K) -> list:
    """
    Embed eval chunks and return the top_k most similar to the hypothesis.
    Each returned chunk has an "embedding" and "score" field added.

    Args:
        eval_chunks: output of get_eval_chunks()
        hypothesis_embedding: np.ndarray from Neo4j h.embedding
        embedder: SentenceTransformer instance
        top_k: number of anchors to return
    """
    texts      = [c["text"] for c in eval_chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False)
    scored = [
        {**chunk, "embedding": emb,
         "score": cosine_similarity(emb, hypothesis_embedding)}
        for chunk, emb in zip(eval_chunks, embeddings)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── Section 6 ─────────────────────────────────────────────────────────────────

def get_precedents(anchors: list, h_id: str, driver,
                   per_label: int = PER_LABEL_K,
                   min_score: float = MIN_SCORE) -> dict:
    """
    Query Neo4j with every anchor, deduplicate, and return the top per_label
    precedents for each label above min_score.

    Args:
        anchors: output of get_anchors()
        h_id: hypothesis ID e.g. "H04"
        driver: neo4j.GraphDatabase driver instance
        per_label: max precedents per label
        min_score: minimum cosine similarity to include (0.0–1.0)

    Returns:
        {"ENTAILED": [...], "CONTRADICTED": [...]}
        Each entry has keys: clause_id, text, score
    """
    seen      = {"ENTAILED": set(), "CONTRADICTED": set()}
    collected = {"ENTAILED": [], "CONTRADICTED": []}

    for anchor in anchors:
        emb_list = anchor["embedding"].tolist()
        with driver.session() as s:
            for label in ("ENTAILED", "CONTRADICTED"):
                rows = s.run(_PRECEDENT_QUERY, {
                    "embedding": emb_list,
                    "h_id":      h_id,
                    "label":     label,
                    "per_label": per_label,
                    "min_score": min_score,
                }).data()
                for row in rows:
                    if row["clause_id"] not in seen[label]:
                        seen[label].add(row["clause_id"])
                        collected[label].append(row)

    for label in collected:
        collected[label] = sorted(
            collected[label], key=lambda x: x["score"], reverse=True
        )[:per_label]

    return collected


# ── Section 7 ─────────────────────────────────────────────────────────────────

def build_prompt(h_id: str, anchors: list, precedents: dict,
                 h_data: dict, max_few_shots: int = MAX_FEW_SHOTS) -> str:
    """
    Build the dynamic few-shot prompt for a single hypothesis.
    Zero-shot fallback when no precedents were retrieved.

    Args:
        h_id: hypothesis ID e.g. "H04"
        anchors: output of get_anchors()
        precedents: output of get_precedents()
        h_data: dict with keys title, definition (fetch from Neo4j or HYPOTHESES)
        max_few_shots: max examples per label shown in the prompt

    Returns:
        Prompt string ready to send to the LLM. Ends with "Verdict: ".
    """
    few_shot_lines = []
    for label, symbol in [("ENTAILED", "ENTAILED"), ("CONTRADICTED", "CONTRADICTED")]:
        for p in precedents[label][:max_few_shots]:
            few_shot_lines.append(
                f"[Example — {symbol}]\n"
                f"Clause: {p['text'].strip()}\n"
                f"Verdict: {symbol}"
            )

    few_shot_block = (
        "### Precedents from similar NDAs\n" + "\n\n".join(few_shot_lines) + "\n"
        if few_shot_lines else ""
    )
    evidence_block = "\n".join(f"  • {a['text'].strip()}" for a in anchors)

    return (
        f"You are a legal NDA analyst. Classify one hypothesis based only on the contract evidence provided.\n\n"
        f"## Hypothesis [{h_id}]: {h_data['title']}\n"
        f"{h_data['definition']}\n\n"
        f"{few_shot_block}"
        f"### Evidence from the contract under review\n"
        f"{evidence_block}\n\n"
        f"Classify the hypothesis as exactly one of: ENTAILED / CONTRADICTED / NOT_MENTIONED\n"
        f"Then provide a one-sentence justification citing specific evidence.\n\n"
        f"Verdict: "
    )


def fetch_hypothesis(h_id: str, driver) -> dict:
    """
    Fetch hypothesis metadata and embedding from Neo4j.
    Returns dict with keys: title, definition, embedding (np.ndarray), criticality.
    """
    with driver.session() as s:
        row = s.run(
            """
            MATCH (h:Hypothesis {h_id: $h_id})
            RETURN h.title       AS title,
                   h.definition  AS definition,
                   h.embedding   AS embedding,
                   h.criticality AS criticality
            """,
            h_id=h_id,
        ).single()
    return {
        "title":       row["title"],
        "definition":  row["definition"],
        "embedding":   np.array(row["embedding"]),
        "criticality": row["criticality"],
    }
