import numpy as np
import pytest

from services.embeddings.embedding_service import EmbeddingService, EmbeddingServiceError, EmbeddingValidationError
from services.embeddings.embedding_cache import SQLiteEmbeddingCache
from services.embeddings.embedding_jobs import JOB_COMPLETED, JOB_FAILED, EmbeddingBatchProcessor, SQLiteEmbeddingJobStore
from retrieval.vector_store import PersistentVectorStore, PGVectorStore, VectorDimensionError, VectorRecord, VectorStoreError
from security.access_control import AccessContext, AccessPolicy
from services.embeddings.embedding_observability import EmbeddingMetrics, StructuredEventLogger
import retrieval.embeddings_rag as embeddings_rag


class FakeEmbeddingModel:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([[float(index + 1)] * self.dimension for index, _ in enumerate(texts)])


def test_embedding_service_batches_and_attaches_model_metadata():
    model = FakeEmbeddingModel(dimension=3)
    service = EmbeddingService(model_name="test-model", model=model, batch_size=2, expected_dimension=3)

    vectors = service.embed_documents(["one", "two", "three"])

    assert len(vectors) == 3
    assert len(vectors[0]) == 3
    assert len(model.calls) == 2
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert service.embed_query("query") == [1.0, 1.0, 1.0]
    assert service.metadata()["embedding_model"] == "test-model"
    assert service.metadata()["embedding_dimension"] == 3
    assert service.health_check()["ready"] is True


def test_embedding_service_rejects_dimension_mismatch_and_bad_inputs():
    with pytest.raises(EmbeddingValidationError, match="dimension mismatch"):
        EmbeddingService(model=FakeEmbeddingModel(dimension=3), expected_dimension=4)

    service = EmbeddingService(model=FakeEmbeddingModel(dimension=3))
    with pytest.raises(EmbeddingValidationError, match="non-empty string"):
        service.embed_query(" ")


def test_embedding_service_rejects_bad_model_output():
    class BadModel(FakeEmbeddingModel):
        def encode(self, texts, **kwargs):
            return np.asarray([[float("nan")] * self.dimension for _ in texts])

    service = EmbeddingService(model=BadModel(dimension=3))
    with pytest.raises(EmbeddingValidationError, match="finite numeric"):
        service.embed_documents(["bad"])


def test_embedding_service_wraps_batch_failures():
    class FailingModel(FakeEmbeddingModel):
        def encode(self, texts, **kwargs):
            raise ValueError("model unavailable")

    service = EmbeddingService(model=FailingModel())
    with pytest.raises(EmbeddingServiceError, match="batch failed"):
        service.embed_documents(["text"])


def test_embedding_service_caches_vectors_by_text_and_model_identity():
    model = FakeEmbeddingModel(dimension=3)
    service = EmbeddingService(model_name="test-model", model=model, batch_size=2, cache_size=10)

    first = service.embed_documents(["same text", "same text", "other text"])
    second = service.embed_documents(["same text", "other text"])

    assert first[0] == first[1]
    assert second == [first[0], first[2]]
    assert len(model.calls) == 1
    assert service.cache_stats() == {
        "enabled": True,
        "size": 2,
        "capacity": 10,
        "hits": 2,
        "misses": 2,
        "persistent_hits": 0,
        "persistent_enabled": False,
    }

    service.clear_cache()
    assert service.cache_stats()["size"] == 0
    assert service.cache_stats()["hits"] == 0


def test_sqlite_cache_survives_service_recreation(tmp_path):
    cache_path = tmp_path / "embeddings.sqlite3"
    first_model = FakeEmbeddingModel(dimension=3)
    first = EmbeddingService(
        model_name="test-model",
        model=first_model,
        persistent_cache=SQLiteEmbeddingCache(cache_path),
    )
    expected = first.embed_documents(["persist me"])[0]

    second_model = FakeEmbeddingModel(dimension=3)
    second = EmbeddingService(
        model_name="test-model",
        model=second_model,
        persistent_cache=SQLiteEmbeddingCache(cache_path),
    )
    actual = second.embed_documents(["persist me"])[0]

    assert actual == expected
    assert second_model.calls == []
    assert second.cache_stats()["persistent_hits"] == 1


def test_embedding_batch_processor_retries_and_checkpoints(tmp_path):
    class FlakyModel(FakeEmbeddingModel):
        def __init__(self):
            super().__init__(dimension=3)
            self.failures_remaining = 1

        def encode(self, texts, **kwargs):
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise ValueError("temporary model failure")
            return super().encode(texts, **kwargs)

    store = SQLiteEmbeddingJobStore(tmp_path / "jobs.sqlite3")
    service = EmbeddingService(model=FlakyModel(), cache_enabled=False)
    processor = EmbeddingBatchProcessor(service, store, batch_size=2, max_retries=2)

    job = processor.submit(["one", "two", "three"])
    vectors = processor.run(job.job_id)
    status = processor.status(job.job_id)

    assert len(vectors) == 3
    assert status.status == JOB_COMPLETED
    assert status.completed_items == 3
    assert status.attempts == 3
    assert processor.resume(job.job_id) == vectors


def test_embedding_batch_processor_records_failed_jobs(tmp_path):
    class AlwaysFailingModel(FakeEmbeddingModel):
        def encode(self, texts, **kwargs):
            raise ValueError("permanent model failure")

    store = SQLiteEmbeddingJobStore(tmp_path / "jobs.sqlite3")
    service = EmbeddingService(model=AlwaysFailingModel(), cache_enabled=False)
    processor = EmbeddingBatchProcessor(service, store, max_retries=1)
    job = processor.submit(["cannot embed"])

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        processor.run(job.job_id)

    status = processor.status(job.job_id)
    assert status.status == JOB_FAILED
    assert status.completed_items == 0
    assert "permanent model failure" in status.last_error


def test_advanced_rag_uses_service_dimension_and_embedding_methods(monkeypatch):
    monkeypatch.setattr(embeddings_rag, "CrossEncoder", lambda _: (_ for _ in ()).throw(RuntimeError("disabled")))
    service = EmbeddingService(model_name="test-model", model=FakeEmbeddingModel(dimension=3))
    rag = embeddings_rag.AdvancedRAG(llm=object(), embedding_service=service)

    rag.add_documents([
        ("first document", {"doc_id": "doc-1"}),
        ("second document", {"doc_id": "doc-2"}),
    ])

    assert rag.embedding_dim == 3
    assert rag._emb_matrix.shape == (2, 3)
    results = rag.hybrid_search("find this", top_k=1)
    assert len(results) == 1


def test_persistent_vector_store_rebuilds_and_reloads_atomically(tmp_path):
    store = PersistentVectorStore(tmp_path / "vectors", model_name="test-model", dimension=3, use_faiss=False)
    manifest = store.upsert([
        VectorRecord("chunk-1", [1.0, 0.0, 0.0], {"document_id": "doc-1"}),
        VectorRecord("chunk-2", [0.0, 1.0, 0.0], {"document_id": "doc-2"}),
    ])

    assert manifest["count"] == 2
    assert store.search([0.9, 0.1, 0.0], top_k=1)[0][0] == "chunk-1"
    assert store.consistency_check()["consistent"] is True
    store.close()

    reloaded = PersistentVectorStore(tmp_path / "vectors", model_name="test-model", dimension=3, use_faiss=False)
    assert reloaded.search([0.0, 0.8, 0.1], top_k=1)[0][0] == "chunk-2"
    assert reloaded.consistency_check()["index_count"] == 2


def test_persistent_vector_store_rejects_invalid_dimensions(tmp_path):
    store = PersistentVectorStore(tmp_path / "vectors", model_name="test-model", dimension=3, use_faiss=False)

    with pytest.raises(VectorDimensionError):
        store.upsert([VectorRecord("bad", [1.0, 0.0], {})])


def test_persistent_vector_store_enforces_tenant_and_metadata_filters(tmp_path):
    store = PersistentVectorStore(
        tmp_path / "tenant-vectors",
        model_name="test-model",
        dimension=3,
        use_faiss=False,
        tenant_id="tenant-a",
    )
    store.upsert([
        VectorRecord("a-1", [1.0, 0.0, 0.0], {"tenant_id": "tenant-a", "department": "hr"}),
        VectorRecord("a-2", [0.9, 0.1, 0.0], {"tenant_id": "tenant-a", "department": "finance"}),
    ])

    filtered = store.search([1.0, 0.0, 0.0], metadata_filter={"department": "finance"})
    assert filtered[0][0] == "a-2"
    assert filtered[0][1] == pytest.approx(0.9)
    with pytest.raises(VectorStoreError, match="tenant"):
        store.upsert([VectorRecord("wrong", [1.0, 0.0, 0.0], {"tenant_id": "tenant-b"})])


def test_access_policy_enforces_clearance_acl_and_ownership():
    policy = AccessPolicy()
    context = AccessContext(
        tenant_id="tenant-a",
        user_id="user-1",
        roles={"finance"},
        departments={"finance"},
        clearance_level="confidential",
    )

    assert policy.can_access({"tenant_id": "tenant-a", "classification": "confidential", "allowed_roles": ["finance"]}, context)
    assert not policy.can_access({"tenant_id": "tenant-a", "classification": "restricted"}, context)
    assert not policy.can_access({"tenant_id": "tenant-b"}, context)
    assert not policy.can_access({"tenant_id": "tenant-a", "allowed_user_ids": ["user-2"]}, context)
    assert not policy.can_access({"tenant_id": "tenant-a", "owner_user_id": "user-2"}, context)
    assert policy.can_access({"tenant_id": "tenant-a", "owner_user_id": "user-2"}, AccessContext("tenant-a", user_id="admin", roles={"admin"}))


def test_persistent_vector_store_applies_access_context(tmp_path):
    store = PersistentVectorStore(
        tmp_path / "secure-vectors",
        model_name="test-model",
        dimension=3,
        use_faiss=False,
        tenant_id="tenant-a",
    )
    store.upsert([
        VectorRecord("public", [1.0, 0.0, 0.0], {"tenant_id": "tenant-a", "classification": "public"}),
        VectorRecord("secret", [0.99, 0.01, 0.0], {"tenant_id": "tenant-a", "classification": "confidential", "allowed_roles": ["finance"]}),
    ])

    hr_context = AccessContext("tenant-a", user_id="hr-user", roles={"hr"}, clearance_level="confidential")
    finance_context = AccessContext("tenant-a", user_id="finance-user", roles={"finance"}, clearance_level="confidential")
    assert [item[0] for item in store.search([1.0, 0.0, 0.0], access_context=hr_context)] == ["public"]
    assert [item[0] for item in store.search([1.0, 0.0, 0.0], access_context=finance_context)] == ["public", "secret"]


def test_advanced_rag_filters_results_by_access_context(monkeypatch):
    monkeypatch.setattr(embeddings_rag, "CrossEncoder", lambda _: (_ for _ in ()).throw(RuntimeError("disabled")))
    rag = embeddings_rag.AdvancedRAG(
        llm=object(),
        embedding_service=EmbeddingService(model=FakeEmbeddingModel(dimension=3)),
    )
    rag.add_documents([
        ("public policy", {"doc_id": "public", "tenant_id": "tenant-a"}),
        ("finance policy", {"doc_id": "finance", "tenant_id": "tenant-a", "allowed_roles": ["finance"]}),
    ])

    results = rag.hybrid_search("policy", access_context=AccessContext("tenant-a", roles={"hr"}))
    assert all(item["meta"]["doc_id"] == "public" for item in results)


def test_embedding_observability_records_metrics_and_events(capsys):
    metrics = EmbeddingMetrics()
    events = StructuredEventLogger()
    service = EmbeddingService(
        model=FakeEmbeddingModel(dimension=3),
        metrics=metrics,
        event_logger=events,
        cache_enabled=False,
    )

    service.embed_documents(["one", "two"])
    snapshot = metrics.snapshot()

    assert snapshot["counters"]["embedding_requests"] == 1
    assert snapshot["counters"]["embedding_batches"] == 1
    assert snapshot["timings"]["embedding_batch_seconds"]["count"] == 1
    event = events.emit("test_event", request_id="req-1")
    assert event["event"] == "test_event"
    assert event["request_id"] == "req-1"


def test_advanced_rag_restores_persistent_vectors_after_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings_rag, "CrossEncoder", lambda _: (_ for _ in ()).throw(RuntimeError("disabled")))
    store_path = tmp_path / "rag-vectors"
    first_store = PersistentVectorStore(store_path, model_name="test-model", dimension=3, use_faiss=False)
    first = embeddings_rag.AdvancedRAG(
        llm=object(),
        embedding_service=EmbeddingService(model_name="test-model", model=FakeEmbeddingModel(dimension=3)),
        persistent_vector_store=first_store,
    )
    first.add_documents([("persistent employee benefits", {"doc_id": "doc-1"})])
    first_store.close()

    second_store = PersistentVectorStore(store_path, model_name="test-model", dimension=3, use_faiss=False)
    second = embeddings_rag.AdvancedRAG(
        llm=object(),
        embedding_service=EmbeddingService(model_name="test-model", model=FakeEmbeddingModel(dimension=3)),
        persistent_vector_store=second_store,
    )

    assert second._texts == ["persistent employee benefits"]
    assert second.hybrid_search("employee", top_k=1)[0]["text"] == "persistent employee benefits"


def test_pgvector_store_requires_database_driver_and_valid_configuration():
    with pytest.raises(VectorStoreError, match="psycopg|pgvector|database connection"):
        PGVectorStore(connection=None, table_name="office_chunks", model_name="test-model", dimension=3)