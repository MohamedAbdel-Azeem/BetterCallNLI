from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        hypothesis_id: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant precedents from the training corpus."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return retrieval mode identifier: 'vector' or 'graphrag'."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the retriever is initialised and ready for queries."""
