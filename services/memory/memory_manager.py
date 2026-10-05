"""
Memory Manager Module

Implements a long-term memory system for microfinance workflows using ChromaDB.
Supports episodic, semantic, and procedural memories, plus vector similarity retrieval,
long-term summaries, and session-level short-term memory.

Example:
    >>> from llm_interface import LLMFactory
    >>> from memory_manager import MemoryManager
    >>> llm = LLMFactory.create('deepseek')
    >>> mem = MemoryManager(llm=llm, persist_directory='memory_store')
    >>> mem.record('tool_executed', 'Sent payment reminder to CUST001', detail={'tool': 'email_send'}, entity_ids=['CUST001'], goal_tag='reminders')
    >>> mem.add_semantic_fact('Interest rate policy is 18% annually', detail={'policy': 'loan_interest'})
    >>> context = mem.get_prompt_context(query='reminder email for client', top_k=4)
    >>> print(context)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import chromadb
except ImportError as exc:
    chromadb = None  # type: ignore

from embeddings_rag import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MemoryRecord:
    record_id: str
    memory_type: str
    event_type: str
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    entity_ids: List[str] = field(default_factory=list)
    goal_tag: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "memory_type": self.memory_type,
            "event_type": self.event_type,
            "summary": self.summary,
            "detail": self.detail,
            "entity_ids": self.entity_ids,
            "goal_tag": self.goal_tag,
            "timestamp": self.timestamp,
        }


# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """
    Long-term memory manager with episodic, semantic, and procedural stores backed by ChromaDB.

    Memory is stored persistently in a vector database and retrieved via semantic similarity.
    """

    MEMORY_TYPES = ["episodic", "semantic", "procedural"]
    COLLECTION_MAP = {
        "episodic": "episodic_memory",
        "semantic": "semantic_memory",
        "procedural": "procedural_memory",
    }
    COMPRESSION_THRESHOLD = 100

    def __init__(
        self,
        llm: Any,
        persist_directory: str = "memory_store",
        embedding_model_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if chromadb is None:
            raise ImportError(
                "chromadb is required for MemoryManager. Install it with 'pip install chromadb'."
            )

        self.llm = llm
        self.session_id = session_id or str(uuid4())
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self._short_term: List[MemoryRecord] = []
        self._entity_index: Dict[str, List[str]] = {}
        self._long_term_summary: str = ""

        # Use new ChromaDB client API (PersistentClient for file-based storage)
        self._client = chromadb.PersistentClient(path=self.persist_directory)

        self._collections: Dict[str, Any] = {}
        for memory_type in self.MEMORY_TYPES:
            self._collections[memory_type] = self._get_or_create_collection(memory_type)

        logger.info(
            f"MemoryManager initialized session_id={self.session_id} with ChromaDB at {self.persist_directory}"
        )

    def record(
        self,
        event_type: str,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        entity_ids: Optional[List[str]] = None,
        goal_tag: str = "",
        memory_type: Optional[str] = None,
    ) -> MemoryRecord:
        detail = detail or {}
        entity_ids = entity_ids or []
        memory_type = memory_type or self._infer_memory_type(event_type)
        if memory_type not in self.MEMORY_TYPES:
            memory_type = "episodic"

        record = MemoryRecord(
            record_id=str(uuid4()),
            memory_type=memory_type,
            event_type=event_type,
            summary=summary,
            detail=detail,
            entity_ids=entity_ids,
            goal_tag=goal_tag,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        self._short_term.append(record)
        self._index_entities(record)
        self._store_vector_record(record)

        if len(self._short_term) >= self.COMPRESSION_THRESHOLD:
            self._compress_short_term()

        logger.info(f"Recorded {memory_type} memory: {summary}")
        return record

    def add_semantic_fact(
        self,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        entity_ids: Optional[List[str]] = None,
        goal_tag: str = "",
    ) -> MemoryRecord:
        return self.record(
            event_type="semantic_fact",
            summary=summary,
            detail=detail,
            entity_ids=entity_ids,
            goal_tag=goal_tag,
            memory_type="semantic",
        )

    def add_procedural_note(
        self,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        goal_tag: str = "",
    ) -> MemoryRecord:
        return self.record(
            event_type="procedural_note",
            summary=summary,
            detail=detail,
            entity_ids=[],
            goal_tag=goal_tag,
            memory_type="procedural",
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        memory_types = memory_types or self.MEMORY_TYPES
        query_embedding = self.embedding_model.embed_query(query)
        hits: List[Dict[str, Any]] = []

        for memory_type in memory_types:
            collection = self._collections.get(memory_type)
            if collection is None or self._collection_size(collection) == 0:
                continue

            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "ids"],
            )

            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            ids = result.get("ids", [[]])[0]

            for doc, metadata, distance, uid in zip(documents, metadatas, distances, ids):
                score = 1.0 / (1.0 + float(distance)) if distance is not None else 1.0
                hits.append(
                    {
                        "memory_type": memory_type,
                        "id": uid,
                        "text": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "score": score,
                    }
                )

        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:top_k]

    def get_prompt_context(
        self,
        query: Optional[str] = None,
        top_k: int = 5,
        max_chars: int = 800,
    ) -> str:
        sections: List[str] = []

        if self._long_term_summary:
            sections.append("[Long-term memory]\n" + self._long_term_summary)

        if query:
            relevant = self.retrieve(query, top_k=top_k)
            if relevant:
                lines = [f"- ({hit['memory_type']}) {hit['text']}" for hit in relevant]
                sections.append("[Relevant memories]\n" + "\n".join(lines))

        if self._short_term:
            recent = self._format_events(self._short_term[-10:])
            sections.append("[Recent session activity]\n" + recent)

        output = "\n\n".join(sections).strip()
        if len(output) > max_chars:
            output = output[:max_chars].strip()
            last_newline = output.rfind("\n")
            if last_newline > max_chars * 0.8:
                output = output[:last_newline]

        return output

    def get_full_context(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "short_term_count": len(self._short_term),
            "long_term_summary": self._long_term_summary,
            "entity_count": len(self._entity_index),
            "collection_sizes": {
                memory_type: self._collection_size(self._collections[memory_type])
                for memory_type in self.MEMORY_TYPES
            },
            "entity_index_sample": {
                entity_id: summaries[-3:]
                for entity_id, summaries in list(self._entity_index.items())[:20]
            },
        }

    def query_entity(self, entity_id: str) -> List[str]:
        return self._entity_index.get(entity_id, [])

    def has_action_been_taken(self, action_description: str, entity_id: str) -> bool:
        action_lower = action_description.lower()
        for summary in self.query_entity(entity_id):
            if action_lower in summary.lower():
                return True
        return False

    def clear(self, clear_persistent: bool = False) -> None:
        self._short_term.clear()
        self._long_term_summary = ""
        self._entity_index.clear()
        logger.info(f"Cleared memory for session {self.session_id}")
        if clear_persistent:
            for collection in self._collections.values():
                collection.delete()
            self._client.persist()
            logger.info("Cleared persistent Chroma memory store")

    def _infer_memory_type(self, event_type: str) -> str:
        normalized = event_type.lower()
        if any(tag in normalized for tag in ["policy", "rate", "profile", "fact", "rule"]):
            return "semantic"
        if any(tag in normalized for tag in ["procedure", "workflow", "best_practice", "heuristic"]):
            return "procedural"
        return "episodic"

    def _index_entities(self, record: MemoryRecord) -> None:
        for entity_id in record.entity_ids:
            self._entity_index.setdefault(entity_id, []).append(record.summary)

    def _format_memory_text(self, record: MemoryRecord) -> str:
        detail_text = json.dumps(record.detail, ensure_ascii=False, indent=0)
        return (
            f"{record.summary}\n"
            f"Type: {record.event_type}\n"
            f"Entities: {record.entity_ids}\n"
            f"Goal: {record.goal_tag}\n"
            f"Details: {detail_text}"
        )

    def _store_vector_record(self, record: MemoryRecord) -> None:
        collection = self._collections[record.memory_type]
        document_text = self._format_memory_text(record)
        embedding = self.embedding_model.embed_documents([document_text])[0]
        metadata = {
            "record_id": record.record_id,
            "memory_type": record.memory_type,
            "event_type": record.event_type,
            "entity_ids": record.entity_ids,
            "goal_tag": record.goal_tag,
            "timestamp": record.timestamp,
        }
        collection.add(
            ids=[record.record_id],
            documents=[document_text],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        self._client.persist()

    def _create_summary_record(self, summary_text: str) -> None:
        record = MemoryRecord(
            record_id=str(uuid4()),
            memory_type="episodic",
            event_type="episodic_summary",
            summary=summary_text,
            detail={"source": "memory_compression"},
            entity_ids=[],
            goal_tag="",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        self._long_term_summary = (
            f"{self._long_term_summary}\n{summary_text}".strip()
            if self._long_term_summary
            else summary_text
        )
        self._store_vector_record(record)

    def _compress_short_term(self) -> None:
        if not self._short_term:
            return

        formatted_events = self._format_events(self._short_term)
        prompt = (
            "Summarize the following agent activity into 4-6 key facts. "
            "Include dates, customer IDs, tools used, and outcomes. "
            "Keep each fact short and specific. Do not add extra explanation.\n\n"
            f"Activity log:\n{formatted_events}"
        )
        try:
            summary = self.llm.generate(prompt, max_tokens=280).strip()
            if summary:
                self._create_summary_record(summary)
                logger.info("Compressed short-term memory into long-term summary")
        except Exception as exc:
            logger.warning(f"Memory compression failed: {exc}")

        self._short_term.clear()

    def _get_or_create_collection(self, memory_type: str) -> Any:
        name = self.COLLECTION_MAP.get(memory_type, memory_type)
        try:
            return self._client.get_collection(name=name)
        except Exception:
            return self._client.create_collection(name=name)

    def _collection_size(self, collection: Any) -> int:
        try:
            return collection.count()
        except Exception:
            return 0

    def _format_events(self, events: List[MemoryRecord]) -> str:
        if not events:
            return "[No events]"

        lines = []
        for index, event in enumerate(events, start=1):
            time_part = event.timestamp.split("T")[1][:8] if "T" in event.timestamp else event.timestamp
            lines.append(f"{index}. [{time_part}] {event.event_type}: {event.summary}")
        return "\n".join(lines)
