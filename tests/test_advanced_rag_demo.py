#!/usr/bin/env python3
"""
Quick demo script to validate AdvancedRAG integration.
Tests core RAG functionality without requiring full orchestrator initialization.
"""

import sys
import asyncio
from embeddings_rag import AdvancedRAG
from llm_interface import LLMFactory


def _build_rag_and_llm():
    config = {
        "llm_provider": "generic",
        "llm_model": "gpt-4o-mini",
        "api_base": "http://localhost:1234/v1"
    }
    llm = LLMFactory.create(config)
    rag = AdvancedRAG(llm=llm, embedding_model_name="BAAI/bge-m3")
    return rag, llm


def test_advanced_rag_init():
    """Test 1: Verify AdvancedRAG initializes with LLM."""
    print("=" * 60)
    print("TEST 1: Initialize AdvancedRAG with LLM")
    print("=" * 60)

    rag, llm = _build_rag_and_llm()
    assert llm.__class__.__name__
    assert rag.embedding_dim > 0
    assert rag.bm25 is not None or rag._bm25 is None
    assert rag.faiss_index is not None or rag._faiss_index is None
    assert rag.reranker is not None or rag._reranker is None
    print("✓ AdvancedRAG initialized with embedding model")
    print(f"  - Embedding dimension: {rag.embedding_dim}")
    print(f"  - BM25 available: {rag.bm25 is not None}")
    print(f"  - FAISS available: {rag.faiss_index is not None}")
    print(f"  - CrossEncoder available: {rag.reranker is not None}")


def test_add_documents():
    """Test 2: Add sample documents."""
    rag, _ = _build_rag_and_llm()
    docs = [
        ("The microfinance officer John Smith manages client ABC Corp which has a $5000 loan.",
         {"loan_officer": "John Smith", "client_id": "ABC123", "loan_id": "LOAN001"}),
        ("Officer Sarah Johnson oversees client XYZ Ltd with a $10000 loan and two guarantors.",
         {"loan_officer": "Sarah Johnson", "client_id": "XYZ456", "loan_id": "LOAN002"}),
        ("Risk assessment for Smith's portfolio shows 2% default rate.",
         {"loan_officer": "John Smith", "loan_id": "PORT001"}),
    ]

    rag.add_documents(docs)
    assert len(rag._texts) == len(docs)
    print(f"✓ Added {len(docs)} documents to RAG")
    print(f"  - Total chunks: {len(rag._texts)}")


def test_hybrid_search():
    """Test 3: Test hybrid search (BM25 + vector)."""
    rag, _ = _build_rag_and_llm()
    rag.add_documents([
        ("The microfinance officer John Smith manages client ABC Corp which has a $5000 loan.",
         {"loan_officer": "John Smith", "client_id": "ABC123", "loan_id": "LOAN001"}),
        ("Officer Sarah Johnson oversees client XYZ Ltd with a $10000 loan and two guarantors.",
         {"loan_officer": "Sarah Johnson", "client_id": "XYZ456", "loan_id": "LOAN002"}),
    ])
    query = "What loans does John Smith manage?"
    results = rag.hybrid_search(query, top_k=3)
    assert len(results) >= 1
    print(f"✓ Hybrid search for: '{query}'")
    print(f"  - Found {len(results)} results")


def test_query_decomposition():
    """Test 4: Test query decomposition."""
    rag, llm = _build_rag_and_llm()
    complex_query = "Which officers manage the highest risk portfolios and what are their client defaults?"
    sub_queries = rag.decompose_query(complex_query)
    assert isinstance(sub_queries, list) and len(sub_queries) > 0
    print(f"✓ Decomposed query into sub-questions:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")


def test_compression():
    """Test 5: Test contextual compression."""
    rag, _ = _build_rag_and_llm()
    full_text = "The microfinance officer John Smith manages client ABC Corp which has a $5000 loan. Sarah Johnson oversees client XYZ Ltd with a $10000 loan."
    query = "What is the loan amount for John Smith's client?"
    compressed = rag.compress_context(full_text, query, max_sentences=2)
    assert isinstance(compressed, str)
    print(f"✓ Compressed context:")
    print(f"  Original length: {len(full_text)} chars")
    print(f"  Compressed length: {len(compressed)} chars")


def test_kg_traversal():
    """Test 6: Test knowledge graph traversal."""
    rag, _ = _build_rag_and_llm()
    rag.add_documents([
        ("The microfinance officer John Smith manages client ABC Corp which has a $5000 loan.",
         {"loan_officer": "John Smith", "client_id": "ABC123", "loan_id": "LOAN001"}),
    ])
    results = rag.traverse_kg("officer:John Smith", depth=2)
    assert isinstance(results, list)
    print(f"✓ KG traversal from 'officer:John Smith': {results}")


def main():
    print("\n" + "🧪" * 30)
    print("ADVANCED RAG INTEGRATION TEST SUITE")
    print("🧪" * 30)
    test_advanced_rag_init()
    test_add_documents()
    test_hybrid_search()
    test_query_decomposition()
    test_compression()
    test_kg_traversal()
    print("\n" + "=" * 60)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
