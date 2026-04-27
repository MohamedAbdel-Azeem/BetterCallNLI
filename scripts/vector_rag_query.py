import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ── Hypothesis Map ────────────────────────────────────────────────────────────

HYPOTHESIS_MAP = {
    "nda-11": {
        "short_description": "No reverse engineering",
        "hypothesis": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information."
    },
    "nda-16": {
        "short_description": "Return of confidential information",
        "hypothesis": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."
    },
    "nda-15": {
        "short_description": "No licensing",
        "hypothesis": "Agreement shall not grant Receiving Party any right to Confidential Information."
    },
    "nda-10": {
        "short_description": "Confidentiality of Agreement",
        "hypothesis": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated."
    },
    "nda-2": {
        "short_description": "None-inclusion of non-technical information",
        "hypothesis": "Confidential Information shall only include technical information."
    },
    "nda-1": {
        "short_description": "Explicit identification",
        "hypothesis": "All Confidential Information shall be expressly identified by the Disclosing Party."
    },
    "nda-19": {
        "short_description": "Survival of obligations",
        "hypothesis": "Some obligations of Agreement may survive termination of Agreement."
    },
    "nda-12": {
        "short_description": "Permissible development of similar information",
        "hypothesis": "Receiving Party may independently develop information similar to Confidential Information."
    },
    "nda-20": {
        "short_description": "Permissible post-agreement possession",
        "hypothesis": "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information."
    },
    "nda-3": {
        "short_description": "Inclusion of verbally conveyed information",
        "hypothesis": "Confidential Information may include verbally conveyed information."
    },
    "nda-18": {
        "short_description": "No solicitation",
        "hypothesis": "Receiving Party shall not solicit some of Disclosing Party's representatives."
    },
    "nda-7": {
        "short_description": "Sharing with third-parties",
        "hypothesis": "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors)."
    },
    "nda-17": {
        "short_description": "Permissible copy",
        "hypothesis": "Receiving Party may create a copy of some Confidential Information in some circumstances."
    },
    "nda-8": {
        "short_description": "Notice on compelled disclosure",
        "hypothesis": "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."
    },
    "nda-13": {
        "short_description": "Permissible acquirement of similar information",
        "hypothesis": "Receiving Party may acquire information similar to Confidential Information from a third party."
    },
    "nda-5": {
        "short_description": "Sharing with employees",
        "hypothesis": "Receiving Party may share some Confidential Information with some of Receiving Party's employees."
    },
    "nda-4": {
        "short_description": "Limited use",
        "hypothesis": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement."
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def _build_base_prompt(contract_text: str, hypothesis_id: str) -> str:
    """
    ── PLACEHOLDER ──
    Replace the body of this function with your actual prompt template.
    The return value is a string that the rest of the pipeline will append
    few-shot examples to.
    """
    hypothesis_info = HYPOTHESIS_MAP[hypothesis_id]
    prompt = (
    f"Classify the hypothesis based on the contract. "
    f"Respond with ONLY valid JSON nothing else:\n"
    f'{{"label": "ENTAILED" | "CONTRADICTED" | "NOT_MENTIONED", "evidence": ["exact quote 1", "exact quote 2"]}}\n'
    f"If label is NOT_MENTIONED, evidence must be [].\n"
    f"Evidence must be copied verbatim from the contract text, word for word. Do not paraphrase or invent.\n"
    f"Contract:\n{contract_text}\n\n"
    f"Hypothesis:\n{hypothesis_info['hypothesis']}\n\n"
    )
    return prompt


def _find_top_spans(
    hypothesis_embedding: np.ndarray,
    span_texts: list[str],
    span_embeddings: np.ndarray,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
) -> list[str]:
    """
    Rank all span texts by cosine similarity to the hypothesis embedding
    and return the top_k that exceed the threshold.
    """
    scored = []
    for i, span_emb in enumerate(span_embeddings):
        sim = _cosine_similarity(hypothesis_embedding, span_emb)
        if sim >= similarity_threshold:
            scored.append((sim, span_texts[i]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


def _query_chroma_for_examples(
    query_texts: list[str],
    hypothesis_id: str,
    collection,
    embedder: SentenceTransformer,
    distance_threshold: float = 0.3,
) -> list[dict]:
    """
    For each query text, search ChromaDB and keep only results whose
    evidence_for contains the given hypothesis_id.
    Returns a de-duplicated list of {text, label} dicts.
    """
    seen_texts = set()
    examples = []

    for query_text in query_texts:
        query_vector = embedder.encode([query_text]).tolist()

        results = collection.query(
            query_embeddings=query_vector,
            n_results=20,           # fetch more so filtering still leaves enough
            include=["documents", "metadatas", "distances"]
        )

        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            if distance > distance_threshold:
                continue

            text     = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            if text in seen_texts:
                continue

            evidence_for = json.loads(metadata["evidence_for"])

            # Keep only if this hypothesis is listed in evidence_for
            matched = next(
                (e for e in evidence_for if e["hypothesis_id"] == hypothesis_id),
                None
            )
            if matched is None:
                continue

            seen_texts.add(text)
            examples.append({
                "text":  text,
                "label": matched["gold_label"]
            })

    return examples


def _format_examples(examples: list[dict]) -> str:
    """Render the retrieved examples as the few-shot block appended to the prompt."""
    if not examples:
        return ""

    lines = [
        "\nAnd here are some examples for this hypothesis, look for similar things:"
    ]
    for idx, ex in enumerate(examples, start=1):
        label = 'Entailment' if ex['label'] == 'ENTAILED' else 'CONTRADICTED' if ex['label'] == 'Contradiction' else 'NOT_MENTIONED'
        lines.append(f"Span {idx}- {ex['text']}")
        lines.append(f"and it Classifies as {label}")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def build_rag_prompt(
    contract_text: str,
    spans: list[list[int]],
    hypothesis_id: str,
    collection,
    embedder: SentenceTransformer, # use all-MiniLM-L6-v2
    top_k_spans: int = 3,
    span_similarity_threshold: float = 0.4,
    chroma_distance_threshold: float = 0.3,
    verbose: bool = False,
) -> str:
    """
    Build a RAG-augmented prompt for NDA clause classification.

    Parameters
    ----------
    contract_text : str
        Full text of the contract being evaluated.
    spans : list of [char_start, char_end]
        Paragraph spans inside contract_text.
    hypothesis_id : str
        One of the nda-* keys (e.g. 'nda-11').
    collection : chromadb.Collection
        The pre-populated ChromaDB collection.
    embedder : SentenceTransformer
        The same embedding model used when storing chunks.
    top_k_spans : int
        How many of the most relevant spans to use as ChromaDB queries.
    span_similarity_threshold : float
        Minimum cosine similarity for a span to be considered relevant.
    chroma_distance_threshold : float
        Maximum cosine distance for a ChromaDB result to be accepted.

    Returns
    -------
    str
        The final prompt with few-shot examples appended.
    """
    if hypothesis_id not in HYPOTHESIS_MAP:
        raise ValueError(
            f"Unknown hypothesis_id '{hypothesis_id}'. "
            f"Valid keys: {list(HYPOTHESIS_MAP.keys())}"
        )

    # 1. Base prompt (your template)
    prompt = _build_base_prompt(contract_text, hypothesis_id)

    # 2. Extract span texts from the contract
    span_texts = [contract_text[s[0]:s[1]] for s in spans]
    if not span_texts:
        return prompt

    # 3. Embed hypothesis + all spans in one batch
    hypothesis_text = HYPOTHESIS_MAP[hypothesis_id]["hypothesis"]
    all_texts       = [hypothesis_text] + span_texts
    all_embeddings  = embedder.encode(all_texts, show_progress_bar=False)

    hypothesis_embedding = all_embeddings[0]
    span_embeddings      = all_embeddings[1:]

    # 4. Find the most hypothesis-relevant spans
    top_spans = _find_top_spans(
        hypothesis_embedding,
        span_texts,
        span_embeddings,
        top_k=top_k_spans,
        similarity_threshold=span_similarity_threshold,
    )

    if verbose:
        print(f"(Before hitting ChromaDB) Top {len(top_spans)} spans relevant to hypothesis '{hypothesis_id}':")
        for span in top_spans:
            print(f"- {span[:100]}...")  # print first 100 chars of each span

    if not top_spans:
        return prompt     # no spans passed the threshold — return prompt as-is

    # 5. Query ChromaDB using those spans, filtered by hypothesis_id
    examples = _query_chroma_for_examples(
        query_texts=top_spans,
        hypothesis_id=hypothesis_id,
        collection=collection,
        embedder=embedder,
        distance_threshold=chroma_distance_threshold,
    )

    if verbose:
        print(f"(After hitting ChromaDB) Retrieved {len(examples)} few-shot examples for hypothesis '{hypothesis_id}'.")

    # 6. Append few-shot examples block to the prompt
    prompt += _format_examples(examples)
    return prompt