"""Persistent vector records and atomic local index management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from access_control import AccessContext, AccessPolicy
from embedding_observability import EmbeddingMetrics, StructuredEventLogger

try:  # pragma: no cover - depends on the environment
    import faiss
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


class VectorStoreError(RuntimeError):
    """Base error for persistent vector storage."""


class VectorDimensionError(VectorStoreError):
    """Raised when vectors do not match the configured model dimension."""


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    vector: List[float]
    metadata: Dict[str, Any]


class PGVectorStore:
    """Postgres+pgvector adapter compatible with the repository interface.

    This implementation avoids depending on a specific SQLAlchemy layer and works with
    a psycopg/psycopg2 connection object while preserving the tenant-aware metadata
    filtering semantics used elsewhere in the project.
    """

    def __init__(
        self,
        connection: Any,
        *,
        table_name: str = "office_chunks",
        model_name: str,
        dimension: int,
        tenant_id: Optional[str] = None,
        metrics: Optional[EmbeddingMetrics] = None,
        event_logger: Optional[StructuredEventLogger] = None,
    ) -> None:
        if connection is None:
            raise VectorStoreError("pgvector database connection is required; pass a live psycopg/psycopg2 connection.")
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.connection = connection
        self.table_name = table_name
        self.model_name = model_name
        self.dimension = dimension
        self.tenant_id = tenant_id
        self.metrics = metrics or EmbeddingMetrics()
        self.event_logger = event_logger or StructuredEventLogger()
        self._ensure_vector_extension()
        self._ensure_table()

    @property
    def record_ids(self) -> List[str]:
        return [record.record_id for record in self.list_records()]

    def upsert(self, records: Sequence[VectorRecord]) -> Dict[str, Any]:
        if not records:
            return {"count": 0, "table": self.table_name, "tenant_id": self.tenant_id}
        prepared = []
        for record in records:
            self._validate_record(record)
            prepared.append(
                (
                    record.record_id,
                    list(record.vector),
                    json.dumps(record.metadata, separators=(",", ":")),
                    record.metadata.get("tenant_id"),
                )
            )

        self.connection.execute(
            f"""
            INSERT INTO {self.table_name} (record_id, embedding, metadata_json, tenant_id, model_name, dimension, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (record_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                metadata_json = EXCLUDED.metadata_json,
                tenant_id = EXCLUDED.tenant_id,
                model_name = EXCLUDED.model_name,
                dimension = EXCLUDED.dimension,
                updated_at = NOW()
            """,
            [
                (record.record_id, list(record.vector), json.dumps(record.metadata, separators=(",", ":")), record.metadata.get("tenant_id"), self.model_name, self.dimension)
                for record in records
            ],
        )
        self.connection.commit()
        self.metrics.increment("vector_records_upserted", len(records))
        self.event_logger.emit("pgvector_records_upserted", count=len(records), tenant_id=self.tenant_id)
        return {"count": len(records), "table": self.table_name, "tenant_id": self.tenant_id}

    def list_records(self) -> List[VectorRecord]:
        self.connection.execute(
            f"SELECT record_id, embedding, metadata_json FROM {self.table_name} WHERE tenant_id = %s ORDER BY record_id",
            (self.tenant_id,) if self.tenant_id is not None else (None,),
        )
        rows = self.connection.fetchall()
        records = []
        for record_id, embedding, metadata_json in rows:
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else (metadata_json or {})
            vector = self._coerce_vector(embedding)
            records.append(VectorRecord(record_id=record_id, vector=vector, metadata=metadata))
        return records

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        access_context: Optional[AccessContext] = None,
        access_policy: Optional[AccessPolicy] = None,
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []
        self._validate_vector(query_vector)
        policy = access_policy or AccessPolicy()

        rows = self._fetch_candidate_rows()
        if self.tenant_id is not None:
            rows = [row for row in rows if row.metadata.get("tenant_id") == self.tenant_id]
        if metadata_filter:
            rows = [row for row in rows if all(row.metadata.get(key) == value for key, value in metadata_filter.items())]
        if access_context is not None:
            rows = [row for row in rows if policy.can_access(row.metadata, access_context)]
        if not rows:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        scores = []
        for row in rows:
            candidate = np.asarray(row.vector, dtype=np.float32)
            cosine = float(np.dot(candidate, query) / max(np.linalg.norm(candidate) * np.linalg.norm(query), 1e-12))
            scores.append((row.record_id, cosine))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        row = self.connection.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        if hasattr(self.connection, "close"):
            self.connection.close()

    def _ensure_vector_extension(self) -> None:
        try:
            self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.connection.commit()
        except Exception:
            pass

    def _ensure_table(self) -> None:
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                record_id TEXT PRIMARY KEY,
                embedding VECTOR({self.dimension}),
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                tenant_id TEXT,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self.connection.commit()

    def _fetch_candidate_rows(self) -> List[VectorRecord]:
        if self.tenant_id is not None:
            self.connection.execute(
                f"SELECT record_id, embedding, metadata_json FROM {self.table_name} WHERE tenant_id = %s ORDER BY record_id",
                (self.tenant_id,),
            )
        else:
            self.connection.execute(f"SELECT record_id, embedding, metadata_json FROM {self.table_name} ORDER BY record_id")
        rows = self.connection.fetchall()
        return [
            VectorRecord(
                record_id=record_id,
                vector=self._coerce_vector(embedding),
                metadata=json.loads(metadata_json) if isinstance(metadata_json, str) else (metadata_json or {}),
            )
            for record_id, embedding, metadata_json in rows
        ]

    def _validate_record(self, record: VectorRecord) -> None:
        self._validate_vector(record.vector)
        if not record.record_id:
            raise VectorStoreError("Vector record ID cannot be empty.")
        if self.tenant_id is not None and record.metadata.get("tenant_id") != self.tenant_id:
            raise VectorStoreError("Vector record tenant does not match the store tenant.")

    def _validate_vector(self, vector: Sequence[float]) -> None:
        array = np.asarray(vector)
        if array.ndim != 1 or array.shape[0] != self.dimension:
            raise VectorDimensionError(f"Expected vector dimension {self.dimension}, received shape {array.shape}.")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise VectorStoreError("Vectors must contain only finite numeric values.")

    def _coerce_vector(self, embedding: Any) -> List[float]:
        if embedding is None:
            return [0.0] * self.dimension
        if isinstance(embedding, (list, tuple)):
            return [float(value) for value in embedding]
        if isinstance(embedding, str):
            try:
                values = json.loads(embedding)
                if isinstance(values, list):
                    return [float(value) for value in values]
            except Exception:
                pass
            try:
                return [float(value) for value in embedding.strip("[]").split(",") if value.strip()]
            except Exception:
                pass
        if hasattr(embedding, "tolist"):
            return [float(value) for value in embedding.tolist()]
        return [float(embedding)] * self.dimension


class SQLiteVectorRepository:
    """Durable vector records and metadata, independent of the search index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_records (
                record_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def upsert(self, record: VectorRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO vector_records (record_id, vector_json, metadata_json, dimension, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                metadata_json = excluded.metadata_json,
                dimension = excluded.dimension,
                updated_at = excluded.updated_at
            """,
            (
                record.record_id,
                json.dumps(record.vector, separators=(",", ":")),
                json.dumps(record.metadata, separators=(",", ":")),
                len(record.vector),
                self._timestamp(),
            ),
        )
        self._connection.commit()

    def upsert_many(self, records: Iterable[VectorRecord]) -> None:
        self._connection.executemany(
            """
            INSERT INTO vector_records (record_id, vector_json, metadata_json, dimension, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                metadata_json = excluded.metadata_json,
                dimension = excluded.dimension,
                updated_at = excluded.updated_at
            """,
            [
                (
                    record.record_id,
                    json.dumps(record.vector, separators=(",", ":")),
                    json.dumps(record.metadata, separators=(",", ":")),
                    len(record.vector),
                    self._timestamp(),
                )
                for record in records
            ],
        )
        self._connection.commit()

    def list_records(self) -> List[VectorRecord]:
        rows = self._connection.execute(
            "SELECT record_id, vector_json, metadata_json FROM vector_records ORDER BY record_id"
        ).fetchall()
        return [
            VectorRecord(record_id, [float(value) for value in json.loads(vector)], json.loads(metadata))
            for record_id, vector, metadata in rows
        ]

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM vector_records").fetchone()[0])

    def close(self) -> None:
        self._connection.close()

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class PersistentVectorStore:
    """Persist vectors and publish complete index versions atomically."""

    def __init__(self, storage_dir: str | Path, *, model_name: str, dimension: int, use_faiss: Optional[bool] = None, tenant_id: Optional[str] = None, metrics: Optional[EmbeddingMetrics] = None, event_logger: Optional[StructuredEventLogger] = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.dimension = dimension
        self.use_faiss = _FAISS_AVAILABLE if use_faiss is None else use_faiss
        self.tenant_id = tenant_id
        self.metrics = metrics or EmbeddingMetrics()
        self.event_logger = event_logger or StructuredEventLogger()
        if self.use_faiss and not _FAISS_AVAILABLE:
            raise VectorStoreError("FAISS was requested but is not installed.")
        self.repository = SQLiteVectorRepository(self.storage_dir / "vectors.sqlite3")
        self._index = None
        self._record_ids: List[str] = []
        self._manifest: Optional[Dict[str, Any]] = None
        self._load_active_index()

    def upsert(self, records: Sequence[VectorRecord]) -> Dict[str, Any]:
        for record in records:
            self._validate_record(record)
        self.repository.upsert_many(records)
        self.metrics.increment("vector_records_upserted", len(records))
        self.event_logger.emit("vector_records_upserted", count=len(records), tenant_id=self.tenant_id)
        return self.rebuild()

    def rebuild(self) -> Dict[str, Any]:
        self.metrics.increment("vector_index_rebuilds")
        records = self.repository.list_records()
        matrix = self._records_matrix(records)
        version = f"index-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
        temporary_dir = self.storage_dir / f".{version}.tmp"
        version_dir = self.storage_dir / version
        temporary_dir.mkdir(parents=True, exist_ok=False)
        index_type = "faiss" if self.use_faiss else "numpy"
        try:
            if self.use_faiss:
                index = faiss.IndexFlatIP(self.dimension)
                if len(matrix):
                    index.add(matrix)
                faiss.write_index(index, str(temporary_dir / "vectors.faiss"))
            else:
                with (temporary_dir / "vectors.npy").open("wb") as handle:
                    np.save(handle, matrix)

            manifest = {
                "version": version,
                "model_name": self.model_name,
                "tenant_id": self.tenant_id,
                "dimension": self.dimension,
                "count": len(records),
                "record_ids": [record.record_id for record in records],
                "index_type": index_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (temporary_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary_dir, version_dir)
            active_temp = self.storage_dir / ".active.json.tmp"
            active_temp.write_text(json.dumps({"version": version}), encoding="utf-8")
            os.replace(active_temp, self.storage_dir / "active.json")
        except Exception:
            if temporary_dir.exists():
                for child in temporary_dir.iterdir():
                    child.unlink()
                temporary_dir.rmdir()
            raise

        self._load_version(version)
        self.event_logger.emit("vector_index_published", version=version, count=len(records), tenant_id=self.tenant_id)
        return manifest

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        access_context: Optional[AccessContext] = None,
        access_policy: Optional[AccessPolicy] = None,
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []
        self.metrics.increment("vector_searches")
        self._validate_vector(query_vector)
        if self._index is None or not self._record_ids:
            return []
        query = np.asarray([query_vector], dtype=np.float32)
        if metadata_filter or access_context is not None:
            policy = access_policy or AccessPolicy()
            records = [
                record for record in self.repository.list_records()
                if (not metadata_filter or all(record.metadata.get(key) == value for key, value in metadata_filter.items()))
            ]
            if self.tenant_id is not None:
                records = [record for record in records if record.metadata.get("tenant_id") == self.tenant_id]
            if access_context is not None:
                records = [record for record in records if policy.can_access(record.metadata, access_context)]
            if not records:
                return []
            matrix = np.asarray([record.vector for record in records], dtype=np.float32)
            scores = np.dot(matrix, query[0])
            indices = np.argsort(scores)[::-1][:top_k]
            return [(records[index].record_id, float(scores[index])) for index in indices]
        if self.use_faiss:
            scores, indices = self._index.search(query, min(top_k, len(self._record_ids)))
            return [
                (self._record_ids[index], float(scores[0][rank]))
                for rank, index in enumerate(indices[0])
                if index >= 0
            ]
        scores = np.dot(self._index, query[0])
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self._record_ids[index], float(scores[index])) for index in indices]

    def consistency_check(self) -> Dict[str, Any]:
        self.metrics.increment("vector_consistency_checks")
        record_count = self.repository.count()
        manifest_count = int((self._manifest or {}).get("count", 0))
        return {
            "consistent": self._manifest is not None and record_count == manifest_count and len(self._record_ids) == manifest_count,
            "repository_count": record_count,
            "index_count": len(self._record_ids),
            "dimension": self.dimension,
            "model_name": self.model_name,
        }

    def close(self) -> None:
        self.repository.close()

    def _load_active_index(self) -> None:
        active_path = self.storage_dir / "active.json"
        if active_path.exists():
            active = json.loads(active_path.read_text(encoding="utf-8"))
            self._load_version(active["version"])

    def _load_version(self, version: str) -> None:
        version_dir = self.storage_dir / version
        manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["model_name"] != self.model_name
            or int(manifest["dimension"]) != self.dimension
            or manifest.get("tenant_id") != self.tenant_id
        ):
            raise VectorStoreError("Persistent index model or dimension does not match the active embedding service.")
        if manifest["index_type"] == "faiss":
            self._index = faiss.read_index(str(version_dir / "vectors.faiss"))
            if self._index.d != self.dimension:
                raise VectorDimensionError("Persistent FAISS index dimension is invalid.")
        else:
            self._index = np.load(version_dir / "vectors.npy")
            if self._index.ndim != 2 or self._index.shape[1] != self.dimension:
                raise VectorDimensionError("Persistent NumPy index dimension is invalid.")
        self._record_ids = list(manifest["record_ids"])
        if len(self._record_ids) != int(manifest["count"]):
            raise VectorStoreError("Persistent index manifest count is invalid.")
        self._manifest = manifest

    def _records_matrix(self, records: Sequence[VectorRecord]) -> np.ndarray:
        if not records:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray([record.vector for record in records], dtype=np.float32)

    def _validate_record(self, record: VectorRecord) -> None:
        self._validate_vector(record.vector)
        if not record.record_id:
            raise VectorStoreError("Vector record ID cannot be empty.")
        if self.tenant_id is not None and record.metadata.get("tenant_id") != self.tenant_id:
            raise VectorStoreError("Vector record tenant does not match the store tenant.")

    def _validate_vector(self, vector: Sequence[float]) -> None:
        array = np.asarray(vector)
        if array.ndim != 1 or array.shape[0] != self.dimension:
            raise VectorDimensionError(f"Expected vector dimension {self.dimension}, received shape {array.shape}.")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise VectorStoreError("Vectors must contain only finite numeric values.")


__all__ = [
    "PersistentVectorStore",
    "PGVectorStore",
    "SQLiteVectorRepository",
    "VectorRecord",
    "VectorStoreError",
    "VectorDimensionError",
]