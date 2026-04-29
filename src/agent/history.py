"""
Conversation history manager.

Stores per-turn data (user message, assistant response, evidence, precedents)
and serialises to / deserialises from JSON for persistence across sessions.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConversationHistory:
    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.turns: List[Dict[str, Any]] = []

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        evidence: Optional[List[Dict]] = None,
        precedents: Optional[List[Dict]] = None,
        retrieval_mode: str = "graphrag",
    ) -> None:
        self.turns.append(
            {
                "turn_id": len(self.turns) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": user_message,
                "assistant": assistant_message,
                "evidence": evidence or [],
                "precedents": precedents or [],
                "retrieval_mode": retrieval_mode,
            }
        )

    def clear(self) -> None:
        self.turns.clear()

    # ── Anthropic API format ──────────────────────────────────────────────────

    def to_anthropic_messages(self) -> List[Dict[str, str]]:
        """Return history as alternating user/assistant dicts for the Claude API."""
        messages = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turn_count": len(self.turns),
            "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationHistory:
        h = cls(session_id=data["session_id"])
        h.created_at = data["created_at"]
        h.turns = data["turns"]
        return h

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> ConversationHistory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.turns)

    def __repr__(self) -> str:
        return f"ConversationHistory(session={self.session_id}, turns={len(self)})"
