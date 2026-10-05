"""
Context Retrieval Module

Extracts and manages semantic search logic with support for local (FAISS) and remote
(Supabase) backends. Provides intelligent filtering, deduplication, and formatting for LLM injection.

Example:
    >>> from embeddings_rag import AdvancedRAG
    >>> from supabase_client import SupabaseClient
    >>> rag = AdvancedRAG(llm=None)
    >>> supabase = SupabaseClient()  # optional
    >>> retriever = ContextRetriever(rag=rag, supabase_client=supabase)
    >>> chunks = retriever.retrieve("loan portfolio analysis", top_k=5)
    >>> prompt_text = retriever.format_for_prompt(chunks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from embeddings_rag import AdvancedRAG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RetrievedChunk:
    """Represents a single retrieved chunk with metadata and scoring."""
    
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # "local", "remote", or "both"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }


# ============================================================================
# CONTEXT RETRIEVER
# ============================================================================

class ContextRetriever:
    """
    Manages semantic search across local FAISS index and optional Supabase backend.
    
    Provides intelligent filtering by document category, result merging and deduplication,
    and formatting for LLM prompt injection. Tracks sync status between local and remote stores.
    
    Attributes:
        rag: EmbeddingsRAG instance for local vector search
        supabase: Optional Supabase client for remote sync
        _last_sync_check: Last time sync status was checked
        _sync_warning: Whether local/remote counts are out of sync
    """
    
    # Category inference keywords for retrieve_for_goal()
    GOAL_CATEGORY_MAP = {
        "loan": "Reporting",
        "portfolio": "Reporting",
        "customer": "Customer Handling",
        "client": "Customer Handling",
        "risk": "Compliance",
        "compliance": "Compliance",
        "overdue": "Collections",
        "collection": "Collections",
    }
    
    # Sync check interval in seconds
    SYNC_CHECK_INTERVAL = 5 * 60  # 5 minutes
    
    def __init__(self, rag: AdvancedRAG, supabase_client: Optional[Any] = None):
        """
        Initialize ContextRetriever.
        
        Args:
            rag: EmbeddingsRAG instance for local embeddings and search
            supabase_client: Optional Supabase client for remote search
        """
        self.rag = rag
        self.supabase = supabase_client
        self._last_sync_check: Optional[datetime] = None
        self._sync_warning: bool = False
        self._local_chunk_count: int = 0
        self._remote_chunk_count: int = 0
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks from local and remote sources, merge, deduplicate, and return.
        
        Steps:
        1. Query local FAISS index (request top_k*2 to account for dedup)
        2. Query remote Supabase if available (handles failures gracefully)
        3. Merge and deduplicate by chunk_id, keeping highest score
        4. Sort by score descending and return top_k results
        5. Passive sync check every 5 minutes
        
        Args:
            query: Search query string
            top_k: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"category": "Reporting"})
        
        Returns:
            List of RetrievedChunk objects sorted by relevance score
        """
        local_results = []
        remote_results = []
        
        # Step 1: LOCAL SEARCH
        try:
            logger.debug(f"Querying local index: query={query}, top_k={top_k * 2}, filters={filters}")
            # Support the new AdvancedRAG.retrieve() API when available
            if hasattr(self.rag, "retrieve"):
                res = self.rag.retrieve(query, top_k=top_k * 2)
                # AdvancedRAG.retrieve returns a dict with 'items' by default
                if isinstance(res, dict):
                    items = res.get("items", [])
                    # normalize to expected shape used by older code
                    local_raw = [
                        {
                            "text": it.get("text"),
                            "meta": it.get("meta", {}),
                            "score": float(it.get("score", it.get("rerank_score", 0.0))),
                        }
                        for it in items
                    ]
                elif isinstance(res, list):
                    local_raw = res
                else:
                    local_raw = []
            else:
                # backward-compatible call
                local_raw = self.rag.query(query, topk=top_k * 2, metadata_filter=filters)

            local_results = local_raw if local_raw else []
            logger.debug(f"Local search returned {len(local_results)} results")
        except Exception as e:
            logger.warning(f"Local search failed: {e}")
            local_results = []
        
        # Step 2: REMOTE SEARCH
        if self.supabase is not None:
            try:
                logger.debug(f"Querying remote index: query={query}, top_k={top_k * 2}, filters={filters}")
                remote_raw = self.supabase.query_documents(
                    query, self.rag.embeddings, top_k=top_k * 2, metadata_filter=filters
                )
                remote_results = remote_raw if remote_raw else []
                logger.debug(f"Remote search returned {len(remote_results)} results")
            except Exception as e:
                logger.warning(f"Supabase search failed, using local only: {e}")
                remote_results = []
        
        # Step 3: MERGE AND DEDUPLICATE
        merged_by_id: Dict[str, Dict[str, Any]] = {}
        
        for item in local_results:
            chunk_id = item.get("meta", {}).get("id") or item.get("meta", {}).get("chunk_id")
            if not chunk_id:
                # Fallback: use filename + text hash
                filename = item.get("meta", {}).get("filename", "unknown")
                chunk_id = f"{filename}-{hash(item.get('text', '')) % 10000}"
            
            merged_by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "text": item.get("text", ""),
                "score": item.get("score", 0.0),
                "metadata": item.get("meta", {}),
                "source": "local",
            }
        
        for item in remote_results:
            chunk_id = item.get("meta", {}).get("id") or item.get("meta", {}).get("chunk_id")
            if not chunk_id:
                filename = item.get("meta", {}).get("filename", "unknown")
                chunk_id = f"{filename}-{hash(item.get('text', '')) % 10000}"
            
            if chunk_id in merged_by_id:
                # Duplicate: keep higher score
                if item.get("score", 0.0) > merged_by_id[chunk_id]["score"]:
                    merged_by_id[chunk_id]["score"] = item.get("score", 0.0)
                    merged_by_id[chunk_id]["source"] = "both"
            else:
                merged_by_id[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": item.get("text", ""),
                    "score": item.get("score", 0.0),
                    "metadata": item.get("meta", {}),
                    "source": "remote",
                }
        
        # Sort by score descending
        sorted_results = sorted(
            merged_by_id.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Take top_k
        final_results = sorted_results[:top_k]
        
        # Convert to RetrievedChunk objects
        chunks = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"],
                source=r["source"],
            )
            for r in final_results
        ]
        
        logger.info(f"retrieve() returned {len(chunks)} chunks (merged from {len(local_results)} local + {len(remote_results)} remote)")
        
        # Step 5: PASSIVE SYNC CHECK
        self._check_sync_status()
        
        return chunks
    
    def retrieve_for_goal(self, goal: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve chunks with automatic category filter inference from goal string.
        
        Keyword matching:
        - "loan" or "portfolio" → "Reporting"
        - "customer" or "client" → "Customer Handling"
        - "risk" or "compliance" → "Compliance"
        - "overdue" or "collection" → "Collections"
        - Otherwise: no filter (search all categories)
        
        Args:
            goal: User's goal or intent string
            top_k: Number of results to return (default: 5)
        
        Returns:
            List of RetrievedChunk objects filtered by inferred category
        """
        inferred_category = None
        goal_lower = goal.lower()
        
        # Infer category from keywords
        for keyword, category in self.GOAL_CATEGORY_MAP.items():
            if keyword in goal_lower:
                inferred_category = category
                break
        
        if inferred_category:
            logger.info(f"Inferred category '{inferred_category}' from goal: {goal}")
            filters = {"category": inferred_category}
        else:
            logger.debug(f"No category inferred from goal: {goal}")
            filters = None
        
        return self.retrieve(goal, top_k=top_k, filters=filters)
    
    def format_for_prompt(self, chunks: List[RetrievedChunk], max_chars: int = 3000) -> str:
        """
        Format retrieved chunks into a clean text block for LLM prompt injection.
        
        Produces output like:
        [Source 1: report.pdf | score: 0.92]
        Loan portfolio details...
        
        [Source 2: portfolio.xlsx | score: 0.87]
        Customer breakdown...
        
        ... (2 more sources truncated)
        
        Args:
            chunks: List of RetrievedChunk objects
            max_chars: Maximum total characters before truncation (default: 3000)
        
        Returns:
            Formatted string ready for prompt injection
        """
        if not chunks:
            return "[No relevant documents found.]"
        
        output = ""
        
        for i, chunk in enumerate(chunks):
            filename = chunk.metadata.get("filename", "unknown")
            header = f"[Source {i + 1}: {filename} | score: {chunk.score:.2f}]"
            
            chunk_block = f"{header}\n{chunk.text}\n\n"
            
            # Check if adding this chunk would exceed max_chars
            if len(output) + len(chunk_block) > max_chars and i > 0:
                remaining = len(chunks) - i
                output += f"... ({remaining} more sources truncated)"
                break
            
            output += chunk_block
        
        return output.strip()
    
    def _check_sync_status(self, force: bool = False) -> None:
        """
        Passively check if local and remote chunk counts are in sync.
        
        Runs every SYNC_CHECK_INTERVAL (5 minutes) to compare local vs remote counts.
        If they differ by more than 2, sets _sync_warning flag.
        Never raises exceptions.
        """
        now = datetime.utcnow()
        
        # Check if enough time has passed since last check (unless forced)
        if not force and self._last_sync_check is not None:
            elapsed = (now - self._last_sync_check).total_seconds()
            if elapsed < self.SYNC_CHECK_INTERVAL:
                return
        
        self._last_sync_check = now
        
        try:
            # Get local count from RAG
            local_count = self.rag._local_count or len(self.rag._texts)
            self._local_chunk_count = local_count
            
            # Get remote count if available
            remote_count = 0
            if self.supabase is not None:
                try:
                    remote_count = self.supabase.count_documents()
                    self._remote_chunk_count = remote_count
                except Exception as e:
                    logger.warning(f"Failed to get remote chunk count: {e}")
                    remote_count = self._remote_chunk_count

            # Update RAG sync tracking and keep retriever state aligned.
            if hasattr(self.rag, 'update_remote_count'):
                self.rag.update_remote_count(remote_count)

            # Check if counts are out of sync (differ by more than 2)
            diff = abs(local_count - remote_count)
            self._sync_warning = diff > 2

            if self._sync_warning:
                logger.warning(
                    f"Sync warning: local={local_count}, remote={remote_count}, diff={diff}"
                )
        except Exception as e:
            logger.debug(f"Error checking sync status: {e}")
    
    @property
    def is_synced(self) -> bool:
        """
        Whether local and remote chunk counts are in sync.
        
        Returns True if counts differ by 2 or less, False otherwise.
        """
        return not self._sync_warning
    
    @property
    def sync_status(self) -> Dict[str, Any]:
        """
        Get detailed sync status for Admin Diagnostics display.
        
        Returns:
            Dict with keys:
            - local_chunks: Number of chunks in local index
            - remote_chunks: Number of chunks in remote index
            - is_synced: Boolean indicating sync health
            - last_check: ISO timestamp of last sync check
            - diff: Absolute difference between counts
        """
        # Force an immediate sync status calculation for admin/status calls
        self._check_sync_status(force=True)
        
        last_check_str = (
            self._last_sync_check.isoformat() if self._last_sync_check else "never"
        )
        
        return {
            "local_chunks": self._local_chunk_count,
            "remote_chunks": self._remote_chunk_count,
            "is_synced": self.is_synced,
            "last_check": last_check_str,
            "diff": abs(self._local_chunk_count - self._remote_chunk_count),
        }
