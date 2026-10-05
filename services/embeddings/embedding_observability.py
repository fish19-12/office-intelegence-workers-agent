"""Metrics and structured events for the embedding pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from threading import Lock
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MetricTiming:
    count: int
    total_seconds: float
    max_seconds: float


class EmbeddingMetrics:
    """Thread-safe counters and timings with exportable snapshots."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._timings: Dict[str, MetricTiming] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            current = self._timings.get(name, MetricTiming(0, 0.0, 0.0))
            self._timings[name] = MetricTiming(
                count=current.count + 1,
                total_seconds=current.total_seconds + seconds,
                max_seconds=max(current.max_seconds, seconds),
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "timings": {
                    name: {
                        "count": timing.count,
                        "total_seconds": timing.total_seconds,
                        "max_seconds": timing.max_seconds,
                    }
                    for name, timing in self._timings.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()


class StructuredEventLogger:
    """Emit JSON events through the standard logging system."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("embedding_pipeline")

    def emit(self, event: str, **fields: Any) -> Dict[str, Any]:
        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self.logger.info(json.dumps(payload, default=str, sort_keys=True))
        return payload


__all__ = ["EmbeddingMetrics", "MetricTiming", "StructuredEventLogger"]