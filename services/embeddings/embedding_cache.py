"""Persistence backends for embedding vectors."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union


class SQLiteEmbeddingCache:
    """Durable local cache for model-specific embedding vectors."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def get(self, cache_key: str) -> Optional[List[float]]:
        row = self._connection.execute(
            "SELECT vector_json FROM embedding_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        self._connection.execute(
            "UPDATE embedding_cache SET last_accessed_at = ? WHERE cache_key = ?",
            (self._timestamp(), cache_key),
        )
        self._connection.commit()
        try:
            value = json.loads(row[0])
            return [float(item) for item in value]
        except (TypeError, ValueError, json.JSONDecodeError):
            self.delete(cache_key)
            return None

    def set(self, cache_key: str, vector: List[float]) -> None:
        timestamp = self._timestamp()
        self._connection.execute(
            """
            INSERT INTO embedding_cache (cache_key, vector_json, created_at, last_accessed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                vector_json = excluded.vector_json,
                last_accessed_at = excluded.last_accessed_at
            """,
            (cache_key, json.dumps(vector, separators=(",", ":")), timestamp, timestamp),
        )
        self._connection.commit()

    def delete(self, cache_key: str) -> None:
        self._connection.execute("DELETE FROM embedding_cache WHERE cache_key = ?", (cache_key,))
        self._connection.commit()

    def clear(self) -> None:
        self._connection.execute("DELETE FROM embedding_cache")
        self._connection.commit()

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._connection.close()

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["SQLiteEmbeddingCache"]