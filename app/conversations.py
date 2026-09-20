"""Conversation titles and summaries backed by LangGraph checkpoints."""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import UUID


class ConversationStore:
    """Persist generated titles without duplicating checkpointed messages."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_titles (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                )
                """
            )

    def get_title(self, thread_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT title FROM conversation_titles WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def save_title(self, thread_id: str, title: str) -> None:
        cleaned = title.strip()[:80]
        if not cleaned:
            return
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO conversation_titles (thread_id, title)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title
                """,
                (thread_id, cleaned),
            )

    def delete(self, thread_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM conversation_titles WHERE thread_id = ?", (thread_id,)
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def public_messages(messages: list[Any]) -> list[dict[str, str]]:
    result = []
    for message in messages:
        role = getattr(message, "type", "")
        if role not in {"human", "ai"}:
            continue
        if role == "ai" and getattr(message, "tool_calls", None):
            continue
        content = _content_to_text(getattr(message, "content", ""))
        if content:
            result.append(
                {"role": "user" if role == "human" else "assistant", "content": content}
            )
    return result


def fallback_title(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            compact = re.sub(r"\s+", " ", message["content"]).strip()
            return compact[:24] or "新对话"
    return "新对话"


def _is_browser_thread(thread_id: str) -> bool:
    try:
        return str(UUID(thread_id)) == thread_id.lower()
    except ValueError:
        return False


def conversation_summaries(checkpointer: Any, store: ConversationStore) -> list[dict]:
    latest = {}
    # ponytail: scans local checkpoints; add a summary index if history grows large.
    for item in checkpointer.list(None):
        thread_id = str(item.config.get("configurable", {}).get("thread_id", ""))
        if not _is_browser_thread(thread_id):
            continue
        timestamp = str(item.checkpoint.get("ts", ""))
        previous = latest.get(thread_id)
        if previous is None or timestamp > previous[0]:
            latest[thread_id] = (timestamp, item.checkpoint)

    summaries = []
    for thread_id, (updated_at, checkpoint) in latest.items():
        messages = public_messages(checkpoint.get("channel_values", {}).get("messages", []))
        if not messages:
            continue
        summaries.append(
            {
                "thread_id": thread_id,
                "title": store.get_title(thread_id) or fallback_title(messages),
                "updated_at": updated_at,
                "message_count": len(messages),
            }
        )
    return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)
