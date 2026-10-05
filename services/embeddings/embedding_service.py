"""Standalone embedding generation and validation service.

This module deliberately does not perform retrieval, indexing, or persistence.
It owns only model loading, batching, vector validation, and embedding metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import OrderedDict
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from embedding_observability import EmbeddingMetrics, StructuredEventLogger


class EmbeddingServiceError(RuntimeError):
    """Base error raised by the embedding service."""


class EmbeddingValidationError(EmbeddingServiceError):
    """Raised when an embedding result violates the service contract."""


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Identity and shape information for the active embedding model."""

    model_name: str
    dimension: int
    normalized: bool
    provider: str = "sentence-transformers"
    model_revision: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
            "embedding_normalized": self.normalized,
            "embedding_provider": self.provider,
            "embedding_model_revision": self.model_revision,
        }


class EmbeddingService:
    """Generate validated embeddings without coupling to retrieval or storage."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        expected_dimension: Optional[int] = None,
        model: Optional[Any] = None,
        provider: str = "sentence-transformers",
        model_revision: Optional[str] = None,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        persistent_cache: Optional[Any] = None,
        metrics: Optional[EmbeddingMetrics] = None,
        event_logger: Optional[StructuredEventLogger] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.provider = provider
        self.model_revision = model_revision
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.persistent_cache = persistent_cache
        self.metrics = metrics or EmbeddingMetrics()
        self.event_logger = event_logger or StructuredEventLogger()
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._persistent_cache_hits = 0
        self._model = model or self._load_model(model_name)
        detected_dimension = self._detect_dimension(self._model)
        if expected_dimension is not None and detected_dimension != expected_dimension:
            raise EmbeddingValidationError(
                f"Embedding dimension mismatch: model returned {detected_dimension}, expected {expected_dimension}."
            )
        self.dimension = detected_dimension

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_name=self.model_name,
            dimension=self.dimension,
            normalized=self.normalize_embeddings,
            provider=self.provider,
            model_revision=self.model_revision,
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts in bounded batches and return validated vectors."""
        if not texts:
            return []
        normalized_texts = [self._validate_text(text) for text in texts]
        self.metrics.increment("embedding_requests")
        self.metrics.increment("embedding_texts_requested", len(normalized_texts))
        self.event_logger.emit(
            "embedding_request_started",
            model_name=self.model_name,
            text_count=len(normalized_texts),
            dimension=self.dimension,
        )
        if not self.cache_enabled or self.cache_size == 0:
            return self._encode_batches(normalized_texts)

        results: List[Optional[List[float]]] = [None] * len(normalized_texts)
        pending_texts: List[str] = []
        pending_keys: List[str] = []
        pending_positions: Dict[str, List[int]] = {}
        for index, text in enumerate(normalized_texts):
            cache_key = self._cache_key(text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self.metrics.increment("embedding_cache_hits")
                self._cache.move_to_end(cache_key)
                results[index] = list(cached)
                continue
            if self.persistent_cache is not None:
                persistent = self.persistent_cache.get(cache_key)
                if persistent is not None:
                    validated = self._validate_vectors([persistent], expected_rows=1)[0]
                    self._persistent_cache_hits += 1
                    self.metrics.increment("embedding_persistent_cache_hits")
                    self._store_memory(cache_key, validated)
                    results[index] = validated
                    continue
            if cache_key in pending_positions:
                pending_positions[cache_key].append(index)
                continue
            pending_positions[cache_key] = [index]
            pending_keys.append(cache_key)
            pending_texts.append(text)

        self._cache_misses += len(pending_texts)
        self.metrics.increment("embedding_cache_misses", len(pending_texts))
        encoded_pending = self._encode_batches(pending_texts)
        for cache_key, vector in zip(pending_keys, encoded_pending):
            self._store_memory(cache_key, vector)
            if self.persistent_cache is not None:
                self.persistent_cache.set(cache_key, list(vector))
            for index in pending_positions[cache_key]:
                results[index] = list(vector)

        return [vector for vector in results if vector is not None]

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache usage and capacity information."""
        return {
            "enabled": self.cache_enabled and self.cache_size > 0,
            "size": len(self._cache),
            "capacity": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "persistent_hits": self._persistent_cache_hits,
            "persistent_enabled": self.persistent_cache is not None,
        }

    def clear_cache(self) -> None:
        """Remove cached vectors and reset cache counters."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._persistent_cache_hits = 0
        if self.persistent_cache is not None:
            self.persistent_cache.clear()

    def _encode_batches(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            started_at = time.perf_counter()
            self.metrics.increment("embedding_batches")
            try:
                encoded = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
            except Exception as exc:
                self.metrics.increment("embedding_failures")
                self.event_logger.emit(
                    "embedding_batch_failed",
                    model_name=self.model_name,
                    batch_offset=start,
                    batch_size=len(batch),
                    error=str(exc),
                )
                raise EmbeddingServiceError(
                    f"Embedding batch failed at offset {start}: {exc}"
                ) from exc
            self.metrics.observe("embedding_batch_seconds", time.perf_counter() - started_at)
            vectors.extend(self._validate_vectors(encoded, expected_rows=len(batch)))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed one query using the same model contract as documents."""
        vectors = self.embed_documents([text])
        return vectors[0]

    def metadata(self) -> Dict[str, Any]:
        """Return metadata suitable for attaching to persisted vectors."""
        return {
            **self.model_info.as_dict(),
            "embedding_created_at": datetime.now(timezone.utc).isoformat(),
        }

    def health_check(self) -> Dict[str, Any]:
        """Report model readiness and shape without generating a vector."""
        return {
            "ready": self._model is not None,
            **self.model_info.as_dict(),
            "batch_size": self.batch_size,
        }

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingServiceError(
                "sentence-transformers is required when no model is injected."
            ) from exc
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - model/network dependent
            raise EmbeddingServiceError(f"Failed to load embedding model '{model_name}'.") from exc

    def _detect_dimension(self, model: Any) -> int:
        dimension = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        if dimension is None:
            raise EmbeddingValidationError("Embedding model does not expose its vector dimension.")
        try:
            dimension = int(dimension)
        except (TypeError, ValueError) as exc:
            raise EmbeddingValidationError("Embedding model returned an invalid vector dimension.") from exc
        if dimension <= 0:
            raise EmbeddingValidationError("Embedding model dimension must be greater than zero.")
        return dimension

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingValidationError("Embedding input text must be a non-empty string.")
        return text

    def _cache_key(self, text: str) -> str:
        identity = "|".join(
            [
                self.model_name,
                self.model_revision or "",
                str(self.dimension),
                str(self.normalize_embeddings),
                text,
            ]
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _store_memory(self, cache_key: str, vector: List[float]) -> None:
        self._cache[cache_key] = list(vector)
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _validate_vectors(self, vectors: Any, *, expected_rows: int) -> List[List[float]]:
        array = np.asarray(vectors)
        if array.ndim == 1 and expected_rows == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape != (expected_rows, self.dimension):
            raise EmbeddingValidationError(
                f"Invalid embedding shape {array.shape}; expected ({expected_rows}, {self.dimension})."
            )
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise EmbeddingValidationError("Embeddings must contain only finite numeric values.")
        return array.astype(np.float32).tolist()


__all__ = [
    "EmbeddingService",
    "EmbeddingModelInfo",
    "EmbeddingServiceError",
    "EmbeddingValidationError",
]