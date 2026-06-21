"""
ContractNLI dataset loader.

Tries kagglehub first; falls back to a caller-supplied local path.
Also provides a helper to wrap raw pasted/uploaded text as a contract dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_contractnli(
    split: str = "test",
    local_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load one split of ContractNLI.

    Args:
        split:      "train", "dev", or "test"
        local_path: directory that contains train.json / dev.json / test.json.
                    When None, kagglehub is used to download the dataset.

    Returns:
        Raw ContractNLI dict with "documents" and "labels" keys.
    """
    if local_path:
        fpath = Path(local_path) / f"{split}.json"
        return json.loads(fpath.read_text(encoding="utf-8"))

    try:
        import kagglehub
        path = kagglehub.dataset_download("seiftarek158/contract-nli")
        fpath = Path(path) / "contract-nli" / f"{split}.json"
        return json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load ContractNLI '{split}' split. "
            "Either pass local_path= or configure kagglehub. "
            f"Original error: {exc}"
        ) from exc


def get_test_contracts(
    local_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all test-split contracts as normalised dicts."""
    data = load_contractnli("test", local_path=local_path)
    return [_normalise(doc) for doc in data["documents"]]


def get_contract_by_id(
    contract_id: str,
    local_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return a single test contract by its string ID, or None if not found."""
    for c in get_test_contracts(local_path=local_path):
        if str(c["id"]) == str(contract_id):
            return c
    return None


def contract_from_text(
    text: str,
    contract_id: str = "user-provided",
) -> Dict[str, Any]:
    """Wrap raw pasted / uploaded text as a minimal contract dict."""
    cleaned = text.strip()
    return {
        "id": contract_id,
        "text": cleaned,
        "spans": [],
        "char_count": len(cleaned),
    }


# ── internal ──────────────────────────────────────────────────────────────────

def _normalise(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc["id"],
        "text": doc["text"],
        "spans": doc.get("spans", []),
        "annotation_sets": doc.get("annotation_sets", []),
        "char_count": len(doc["text"]),
    }
