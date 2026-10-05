from typing import Dict, List, Optional, Tuple, Any
import os
import re
import json
from datetime import datetime
from uuid import uuid4

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from embedding_service import EmbeddingService
from access_control import AccessContext, AccessPolicy
from vector_store import PersistentVectorStore, VectorRecord

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "500"))


class HuggingFaceEmbeddings(EmbeddingService):
    """Lightweight wrapper around SentenceTransformer embeddings used in this repo."""

    def __init__(self, model_name: Optional[str] = None, model: Optional[SentenceTransformer] = None):
        super().__init__(model_name=model_name or EMBEDDING_MODEL, model=model)


class AdvancedRAG:
    """
    Advanced RAG: hybrid vector + BM25 retrieval, cross-encoder re-ranking,
    query decomposition, contextual compression, simple knowledge graph, and self-RAG decisioning.

    Usage:
        rag = AdvancedRAG(llm=your_llm)
        rag.add_documents([(text, meta), ...])
        results = rag.retrieve(query)
    """

    def __init__(
        self,
        llm: Any,
        embedding_model_name: Optional[str] = None,
        rerank_model: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        persistent_vector_store: Optional[PersistentVectorStore] = None,
        access_policy: Optional[AccessPolicy] = None,
    ):
        self.llm = llm
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        self.embedding_service = embedding_service or HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.embeddings = self.embedding_service
        self.embedding_dim = self.embedding_service.dimension
        self.persistent_vector_store = persistent_vector_store
        self.access_policy = access_policy or AccessPolicy()
        if self.persistent_vector_store is not None and self.persistent_vector_store.dimension != self.embedding_dim:
            raise ValueError("Persistent vector store dimension does not match the embedding service.")

        # optional reranker
        self.rerank_model_name = rerank_model or RERANK_MODEL
        self._reranker = None
        try:
            self._reranker = CrossEncoder(self.rerank_model_name)
        except Exception:
            self._reranker = None
        self.reranker = self._reranker

        # BM25 structures
        self._texts: List[str] = []
        self._tokenized_texts: List[List[str]] = []
        self._metas: List[dict] = []
        self._chunk_ids: List[str] = []
        self._bm25 = BM25Okapi(self._tokenized_texts) if _BM25_AVAILABLE else None
        self.bm25 = self._bm25

        # vector index
        self._emb_matrix: np.ndarray = np.empty((0, self.embedding_dim), dtype=np.float32)
        self._faiss_index = None
        self.faiss_index = self._faiss_index

        # simple in-memory knowledge graph (adjacency)
        self._kg_nodes: Dict[str, dict] = {}
        self._kg_edges: Dict[str, List[Tuple[str, str]]] = {}

        self._dirty_count = 0
        self._restore_persistent_vectors()

    # -------------------- Document ingestion --------------------
    def add_documents(self, text_meta_pairs: List[Tuple[str, dict]]) -> None:
        """Add document chunks and update BM25 + vector indexes.

        meta should include stable doc_id if you want persistence.
        """
        if not text_meta_pairs:
            return

        texts = [t for t, _ in text_meta_pairs]
        metas = [m for _, m in text_meta_pairs]
        if self.persistent_vector_store is not None and self.persistent_vector_store.tenant_id is not None:
            expected_tenant = self.persistent_vector_store.tenant_id
            if any(meta.get("tenant_id") != expected_tenant for meta in metas):
                raise ValueError("Document metadata tenant does not match the persistent vector store tenant.")
        new_ids = [str(m.get("doc_id", str(uuid4()))) + f"_{i}" for i, m in enumerate(metas)]

        # tokenization for BM25 (very simple whitespace tokenization)
        tokenized = [self._simple_tokenize(t) for t in texts]
        self._tokenized_texts.extend(tokenized)
        self._texts.extend(texts)
        self._metas.extend(metas)
        self._chunk_ids.extend(new_ids)

        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._tokenized_texts)

        # embeddings
        new_embs = np.asarray(self.embedding_service.embed_documents(texts), dtype=np.float32)
        self._emb_matrix = np.vstack([self._emb_matrix, new_embs]) if self._emb_matrix.size else new_embs
        self._dirty_count += len(texts)

        if self.persistent_vector_store is not None:
            self.persistent_vector_store.upsert(
                [
                    VectorRecord(
                        record_id=chunk_id,
                        vector=vector.tolist(),
                        metadata={**meta, "_embedding_text": text},
                    )
                    for chunk_id, vector, text, meta in zip(new_ids, new_embs, texts, metas)
                ]
            )

        # update faiss
        if _FAISS_AVAILABLE:
            if self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                if len(self._emb_matrix) > 0:
                    self._faiss_index.add(self._emb_matrix)
            else:
                self._faiss_index.add(new_embs)
        self.faiss_index = self._faiss_index

        # optionally extract KG entities from metas
        for meta in metas:
            self._ingest_meta_to_kg(meta)

        # compact occasionally
        if self._dirty_count >= COMPACTION_THRESHOLD:
            self.compact()

    def compact(self) -> None:
        """Rebuild indexes if needed (simple rebuilding for FAISS/BM25)."""
        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._tokenized_texts)
        if _FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            if len(self._emb_matrix) > 0:
                self._faiss_index.add(self._emb_matrix)
        self.faiss_index = self._faiss_index
        self._dirty_count = 0

    def _restore_persistent_vectors(self) -> None:
        if self.persistent_vector_store is None:
            return
        records = self.persistent_vector_store.repository.list_records()
        if not records:
            return
        restored = []
        for record in records:
            metadata = dict(record.metadata)
            text = metadata.pop("_embedding_text", None)
            if not isinstance(text, str) or not text:
                continue
            restored.append((record.record_id, text, metadata, record.vector))
        if not restored:
            return
        self._chunk_ids = [item[0] for item in restored]
        self._texts = [item[1] for item in restored]
        self._metas = [item[2] for item in restored]
        self._tokenized_texts = [self._simple_tokenize(text) for text in self._texts]
        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._tokenized_texts)
            self.bm25 = self._bm25
        self._emb_matrix = np.asarray([item[3] for item in restored], dtype=np.float32)
        self.compact()

    # -------------------- Hybrid search --------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
        access_context: Optional[AccessContext] = None,
    ) -> List[dict]:
        """Combine BM25 + vector search results, deduplicate, and return candidates.

        Returns list of dicts: {'text','meta','chunk_id','score'}
        """
        candidates: Dict[str, dict] = {}

        # BM25
        if _BM25_AVAILABLE and self._bm25 is not None:
            tokens = self._simple_tokenize(query)
            bm25_scores = self._bm25.get_scores(tokens)
            bm25_idx = np.argsort(bm25_scores)[::-1][: top_k]
            for i in bm25_idx:
                cid = self._chunk_ids[i]
                score = float(bm25_scores[i])
                candidates[cid] = {"text": self._texts[i], "meta": self._metas[i], "chunk_id": cid, "score": score, "source": "bm25"}

        # Vector search
        if len(self._texts) > 0:
            q_emb = np.asarray([self.embedding_service.embed_query(query)], dtype=np.float32)
            if _FAISS_AVAILABLE and self._faiss_index is not None:
                search_k = min(max(top_k * 3, 50), len(self._texts))
                distances, indices = self._faiss_index.search(q_emb, search_k)
                for rank, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self._texts):
                        continue
                    cid = self._chunk_ids[idx]
                    score = 1.0 / (1.0 + float(distances[0][rank]))
                    # merge with BM25 candidate if exists
                    prev = candidates.get(cid)
                    if prev:
                        prev["score"] = max(prev["score"], score)
                        prev.setdefault("source", "bm25+vector")
                    else:
                        candidates[cid] = {"text": self._texts[idx], "meta": self._metas[idx], "chunk_id": cid, "score": score, "source": "vector"}
            else:
                sims = np.dot(self._emb_matrix, q_emb.T).flatten()
                top_idx = np.argsort(sims)[::-1][: top_k * 3]
                for idx in top_idx:
                    cid = self._chunk_ids[idx]
                    score = float(sims[idx])
                    if cid in candidates:
                        candidates[cid]["score"] = max(candidates[cid]["score"], score)
                    else:
                        candidates[cid] = {"text": self._texts[idx], "meta": self._metas[idx], "chunk_id": cid, "score": score, "source": "vector"}

        # metadata filter
        if metadata_filter:
            candidates = {k: v for k, v in candidates.items() if all(v["meta"].get(kk) == vv for kk, vv in metadata_filter.items())}
        if access_context is not None:
            candidates = {
                key: value
                for key, value in candidates.items()
                if self.access_policy.can_access(value["meta"], access_context)
            }

        # return top_k sorted by score
        sorted_cands = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return sorted_cands[: top_k]

    # -------------------- Re-ranking --------------------
    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """Re-rank a candidate set using cross-encoder if available, otherwise fallback to embedding similarity."""
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        pairs = [[query, t] for t in texts]
        scores: List[float] = []
        if self._reranker is not None:
            try:
                scores = self._reranker.predict(pairs).tolist()
            except Exception:
                scores = []

        if not scores:
            # fallback to embedding dot-product
            q_emb = np.asarray([self.embedding_service.embed_query(query)], dtype=np.float32)
            doc_embs = np.asarray(self.embedding_service.embed_documents(texts), dtype=np.float32)
            sims = (doc_embs @ q_emb.T).flatten()
            scores = [float(s) for s in sims]

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    # -------------------- Query decomposition --------------------
    def decompose_query(self, query: str) -> List[str]:
        """Use the LLM to decompose a complex question into sub-questions.

        Expects LLM to return a JSON array of strings; falls back to simple heuristics.
        """
        prompt = (
            f"Decompose the following user question into a small list of focused sub-questions. "
            f"Return ONLY a JSON array of strings.\nQuestion: {query}"
        )
        try:
            if hasattr(self.llm, "generate_json"):
                parts = self.llm.generate_json(prompt)
            else:
                raw = self.llm.generate(prompt)
                parts = json.loads(raw)
            if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                return parts
        except Exception:
            pass

        # fallback: split on conjunctions and clauses
        clauses = re.split(r"\band\b|\bor\b|\,|\;", query)
        clauses = [c.strip() for c in clauses if len(c.strip()) > 8]
        return clauses[:5] if clauses else [query]

    # -------------------- Contextual compression --------------------
    def compress_context(self, text: str, query: str, max_sentences: int = 3) -> str:
        """Return the most relevant sentences from text that answer query.

        Simple algorithm: split into sentences and score by embedding similarity.
        """
        sents = self._split_sentences(text)
        if not sents:
            return text

        q_emb = np.asarray([self.embedding_service.embed_query(query)], dtype=np.float32)
        sent_embs = np.asarray(self.embedding_service.embed_documents(sents), dtype=np.float32)
        sims = (sent_embs @ q_emb.T).flatten()
        top_idx = np.argsort(sims)[::-1][:max_sentences]
        selected = [sents[i] for i in top_idx]
        return " \n".join(selected)

    # -------------------- Knowledge Graph (simple) --------------------
    def _ingest_meta_to_kg(self, meta: dict) -> None:
        """Ingest structured metadata into a simple graph. Expected keys: loan_officer, client_id, loan_id."""
        # nodes are strings; store node metadata
        if not meta:
            return
        # link LoanOfficer -> manages -> Client -> has -> LoanApplication
        lo = meta.get("loan_officer")
        client = meta.get("client_id")
        loan = meta.get("loan_id")
        if lo:
            self._kg_nodes.setdefault(f"officer:{lo}", {}).update({"type": "loan_officer", "id": lo})
        if client:
            self._kg_nodes.setdefault(f"client:{client}", {}).update({"type": "client", "id": client})
        if loan:
            self._kg_nodes.setdefault(f"loan:{loan}", {}).update({"type": "loan", "id": loan})

        if lo and client:
            self._kg_edges.setdefault(f"officer:{lo}", []).append(("manages", f"client:{client}"))
        if client and loan:
            self._kg_edges.setdefault(f"client:{client}", []).append(("has", f"loan:{loan}"))

    def traverse_kg(self, start: str, depth: int = 2) -> List[Tuple[str, str, str]]:
        """Traverse KG from a node prefix (e.g., 'officer:JOHN') and return edges discovered."""
        edges = []
        seen = set()
        frontier = [(start, 0)]
        while frontier:
            node, d = frontier.pop(0)
            if d >= depth:
                continue
            for rel, tgt in self._kg_edges.get(node, []):
                if (node, rel, tgt) in seen:
                    continue
                edges.append((node, rel, tgt))
                seen.add((node, rel, tgt))
                frontier.append((tgt, d + 1))
        return edges

    # -------------------- Self-RAG decision --------------------
    def should_search(self, query: str) -> bool:
        """Decide whether a retrieval is necessary. Use LLM confidence or simple heuristics.

        If the LLM can answer from memory or the question is trivial, return False.
        Otherwise True.
        """
        prompt = (
            f"Decide if the following question requires searching documents to answer accurately. Reply with JSON: {json.dumps({'search': True})} or {json.dumps({'search': False})}.\nQuestion: {query}"
        )
        try:
            if hasattr(self.llm, "generate_json"):
                parsed = self.llm.generate_json(prompt)
            else:
                raw = self.llm.generate(prompt)
                parsed = json.loads(raw)
            if isinstance(parsed, dict) and "search" in parsed:
                return bool(parsed["search"])
        except Exception:
            pass
        # fallback heuristics: if query contains timeframe, compare, trend, list, return True
        keywords = ["trend", "compare", "over the last", "repayment", "default", "rate", "which", "who"]
        if any(k in query.lower() for k in keywords):
            return True
        return False

    # -------------------- High-level retrieval --------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_self_rag: bool = True,
        access_context: Optional[AccessContext] = None,
    ) -> dict:
        """Main retrieval entry point implementing decomposition, hybrid search, reranking, compression, and KG traversal."""
        result = {"query": query, "items": [], "subqueries": [], "kg_edges": []}

        if use_self_rag and not self.should_search(query):
            # Try to answer directly via LLM without searching
            prompt = f"Answer concisely: {query}\nIf you lack necessary facts, respond with 'UNKNOWN'."
            ans = self.llm.generate(prompt)
            if ans and "UNKNOWN" not in ans:
                result["items"] = [{"text": ans, "meta": {}, "source": "llm_direct"}]
                return result

        # Decompose
        subqs = self.decompose_query(query)
        result["subqueries"] = subqs

        all_candidates = []
        for sq in subqs:
            cands = self.hybrid_search(sq, top_k=20, access_context=access_context)
            reranked = self.rerank(sq, cands, top_k=5)
            # compress
            compressed = []
            for c in reranked:
                snippet = self.compress_context(c["text"], sq, max_sentences=3)
                compressed.append({"text": snippet, "meta": c.get("meta", {}), "chunk_id": c.get("chunk_id"), "score": c.get("rerank_score")})
            all_candidates.extend(compressed)

        # Deduplicate by chunk_id and keep best score
        seen = {}
        for it in all_candidates:
            cid = it.get("chunk_id")
            if cid not in seen or it.get("score", 0) > seen[cid]["score"]:
                seen[cid] = it

        items = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        result["items"] = items

        # Simple KG traversal: if query mentions an officer, return edges
        m = re.search(r"officer[: ]?(\w+)", query, re.IGNORECASE)
        if m:
            node = f"officer:{m.group(1)}"
            result["kg_edges"] = self.traverse_kg(node, depth=3)

        return result

    # -------------------- Utilities --------------------
    def _simple_tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\w+", text)]

    def _split_sentences(self, text: str) -> List[str]:
        # naive sentence splitter
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

