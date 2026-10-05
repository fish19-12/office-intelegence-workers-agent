"""Durable, queue-agnostic embedding batch jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from embedding_service import EmbeddingService
from embedding_observability import EmbeddingMetrics, StructuredEventLogger


JOB_PENDING = "pending"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"


@dataclass(frozen=True)
class EmbeddingJob:
    job_id: str
    status: str
    total_items: int
    completed_items: int
    attempts: int
    last_error: Optional[str]
    created_at: str
    updated_at: str


class SQLiteEmbeddingJobStore:
    """SQLite state store for resumable embedding jobs and item checkpoints."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total_items INTEGER NOT NULL,
                completed_items INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_job_items (
                job_id TEXT NOT NULL,
                item_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT,
                status TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (job_id, item_index),
                FOREIGN KEY (job_id) REFERENCES embedding_jobs(job_id)
            )
            """
        )
        self._connection.commit()

    def create(self, texts: Sequence[str], job_id: Optional[str] = None) -> EmbeddingJob:
        identifier = job_id or str(uuid4())
        now = self._timestamp()
        self._connection.execute(
            "INSERT INTO embedding_jobs (job_id, status, total_items, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (identifier, JOB_PENDING, len(texts), now, now),
        )
        self._connection.executemany(
            "INSERT INTO embedding_job_items (job_id, item_index, text, status) VALUES (?, ?, ?, ?)",
            [(identifier, index, text, JOB_PENDING) for index, text in enumerate(texts)],
        )
        self._connection.commit()
        return self.get(identifier)

    def get(self, job_id: str) -> EmbeddingJob:
        row = self._connection.execute(
            "SELECT job_id, status, total_items, completed_items, attempts, last_error, created_at, updated_at FROM embedding_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown embedding job: {job_id}")
        return EmbeddingJob(*row)

    def update(self, job_id: str, *, status: Optional[str] = None, completed_items: Optional[int] = None, attempts: Optional[int] = None, last_error: Optional[str] = None) -> EmbeddingJob:
        current = self.get(job_id)
        values = (
            status if status is not None else current.status,
            completed_items if completed_items is not None else current.completed_items,
            attempts if attempts is not None else current.attempts,
            last_error,
            self._timestamp(),
            job_id,
        )
        self._connection.execute(
            "UPDATE embedding_jobs SET status = ?, completed_items = ?, attempts = ?, last_error = ?, updated_at = ? WHERE job_id = ?",
            values,
        )
        self._connection.commit()
        return self.get(job_id)

    def pending_items(self, job_id: str) -> List[tuple[int, str]]:
        rows = self._connection.execute(
            "SELECT item_index, text FROM embedding_job_items WHERE job_id = ? AND status != ? ORDER BY item_index",
            (job_id, JOB_COMPLETED),
        ).fetchall()
        return [(int(index), text) for index, text in rows]

    def mark_completed(self, job_id: str, items: Sequence[tuple[int, List[float]]]) -> None:
        self._connection.executemany(
            "UPDATE embedding_job_items SET status = ?, vector_json = ?, error = NULL WHERE job_id = ? AND item_index = ?",
            [(JOB_COMPLETED, json.dumps(vector, separators=(",", ":")), job_id, index) for index, vector in items],
        )
        completed = self._connection.execute(
            "SELECT COUNT(*) FROM embedding_job_items WHERE job_id = ? AND status = ?",
            (job_id, JOB_COMPLETED),
        ).fetchone()[0]
        self._connection.execute(
            "UPDATE embedding_jobs SET completed_items = ?, updated_at = ? WHERE job_id = ?",
            (completed, self._timestamp(), job_id),
        )
        self._connection.commit()

    def vectors(self, job_id: str) -> List[List[float]]:
        rows = self._connection.execute(
            "SELECT vector_json FROM embedding_job_items WHERE job_id = ? ORDER BY item_index",
            (job_id,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        self._connection.close()

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class EmbeddingBatchProcessor:
    """Run embedding jobs in retryable, checkpointed batches."""

    def __init__(self, service: EmbeddingService, store: SQLiteEmbeddingJobStore, *, batch_size: int = 32, max_retries: int = 3, metrics: Optional[EmbeddingMetrics] = None, event_logger: Optional[StructuredEventLogger] = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.service = service
        self.store = store
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.metrics = metrics or service.metrics
        self.event_logger = event_logger or service.event_logger

    def submit(self, texts: Sequence[str], job_id: Optional[str] = None) -> EmbeddingJob:
        job = self.store.create(texts, job_id=job_id)
        self.metrics.increment("embedding_jobs_submitted")
        self.event_logger.emit("embedding_job_submitted", job_id=job.job_id, item_count=job.total_items)
        return job

    def run(self, job_id: str) -> List[List[float]]:
        job = self.store.get(job_id)
        self.store.update(job_id, status=JOB_RUNNING, last_error=None)
        pending = self.store.pending_items(job_id)
        try:
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start : start + self.batch_size]
                vectors = self._embed_with_retries(job_id, [text for _, text in batch])
                self.store.mark_completed(job_id, list(zip([index for index, _ in batch], vectors)))
            self.store.update(job_id, status=JOB_COMPLETED, last_error=None)
            self.metrics.increment("embedding_jobs_completed")
            self.event_logger.emit("embedding_job_completed", job_id=job_id, item_count=len(self.store.vectors(job_id)))
            return self.store.vectors(job_id)
        except Exception as exc:
            self.store.update(job_id, status=JOB_FAILED, last_error=str(exc))
            self.metrics.increment("embedding_jobs_failed")
            self.event_logger.emit("embedding_job_failed", job_id=job_id, error=str(exc))
            raise

    def resume(self, job_id: str) -> List[List[float]]:
        return self.run(job_id)

    def status(self, job_id: str) -> EmbeddingJob:
        return self.store.get(job_id)

    def _embed_with_retries(self, job_id: str, texts: Sequence[str]) -> List[List[float]]:
        attempts = self.store.get(job_id).attempts
        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            attempts += 1
            self.store.update(job_id, attempts=attempts)
            if attempts > 1:
                self.metrics.increment("embedding_batch_retries")
            try:
                return self.service.embed_documents(texts)
            except Exception as exc:
                last_error = exc
                self.store.update(job_id, last_error=str(exc))
        raise RuntimeError(f"Embedding batch failed after {self.max_retries + 1} attempts: {last_error}") from last_error


__all__ = [
    "EmbeddingBatchProcessor",
    "EmbeddingJob",
    "SQLiteEmbeddingJobStore",
    "JOB_PENDING",
    "JOB_RUNNING",
    "JOB_COMPLETED",
    "JOB_FAILED",
]