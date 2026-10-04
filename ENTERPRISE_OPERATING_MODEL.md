# Enterprise Operating Model

## Runtime topology

The deployed request path is:

`Render -> office_intelligence.api -> app.asgi_app -> backend_api.app`

`/health/live` is a cheap process liveness probe. The Render entrypoint's `/health/ready` checks that the shared secret and allowed-origins setting are present and that `LLM_PROVIDER` is not `mock`; it does not validate provider credentials/connectivity or reject wildcard origins. All business routes require an `Authorization: Bearer <service-token>` header. Service tokens are HMAC-SHA256 signed and must contain issuer `denbegaye-nextjs`, audience `office-intelligence`, `iat`, and `exp` claims.

For production, configure `OFFICE_INTELLIGENCE_SHARED_SECRET`, `OFFICE_INTELLIGENCE_ALLOWED_ORIGINS` with explicit origins, and a real `LLM_PROVIDER`. Keep `LLM_PROVIDER=mock` only for local development or tests.

## Data ownership

- Uploaded source files live under `UPLOAD_DIR` and are treated as untrusted input.
- `DocumentManager` parses, chunks, enriches, and indexes documents into the local `AdvancedRAG` index.
- When Supabase is configured, document chunks and embeddings are written to the `documents` table.
- `MemoryManager` stores episodic, semantic, and procedural memory in persistent ChromaDB under `memory_store`.
- Session summaries, tool calls, and code execution records are written to Supabase audit tables when those calls are made.
- Tools read credentials from environment variables first and the local credential file only as a development fallback. Production credentials belong in the hosting provider's secret manager.
- `ToolCredentialStore` loads `tool_credentials.json` and merges its values over environment variables, so file values can override environment values. The file is plaintext, process-local, and not tenant-scoped. Do not treat it as a development-only fallback or a production secret store; replace it with an encrypted, access-controlled, tenant-aware store before customer use.

## Current synchronization behavior

Document ingestion currently uses a write-through, best-effort flow:

1. Parse and chunk the source file.
2. Add chunks to the local vector index.
3. Upsert the same chunks and embeddings to Supabase.
4. Return `_supabase_synced=false` when the remote write fails.

This is not yet a distributed transaction. A local write can succeed while the remote write fails, and the in-memory RAG index is not shared between multiple web instances. Supabase retrieval is currently a full-table scan in Python, so it is suitable for small data volumes but not enterprise scale.

## Enterprise synchronization target

The next production stage should use Supabase/Postgres as the system of record and a durable outbox:

1. Store a document manifest with tenant, owner, content hash, version, status, and retention metadata.
2. Store each chunk with `(document_id, version, chunk_index)` as an idempotency key.
3. In the same database transaction, write the manifest and an outbox event such as `document.index.requested`.
4. A worker claims outbox rows with leases, computes embeddings, and upserts chunks using conflict-safe keys.
5. The worker records attempts, errors, and `synced_at`; retries use exponential backoff and a dead-letter state.
6. Readers query the remote vector index by tenant and ACL filters. Local caches are disposable and rebuilt from the system of record.
7. Deletion creates a tombstone/outbox event first, then removes all versions and derived embeddings after retention policy checks.

This gives at-least-once delivery with idempotent consumers. Exactly-once processing should not be assumed; idempotency and version checks provide the correctness guarantee.

## Required enterprise controls

- Tenant and user identity must be part of every document, memory, audit, and retrieval filter.
- Enforce authorization before retrieval, download, tool execution, and report generation; authentication alone is insufficient.
- Replace filename-based uploads with generated object keys, content hashes, malware scanning, and object storage.
- Move ChromaDB and local upload persistence off ephemeral web instances.
- Add Postgres vector indexes/RPC retrieval rather than fetching every embedding into Python.
- Put long-running ingestion, OCR, embedding, and report work on a durable queue.
- Add rate limits, request IDs, idempotency keys, timeouts, circuit breakers, and bounded concurrency.
- Redact secrets and sensitive payloads from audit logs; define retention and deletion policies.
- Add metrics and traces for request latency, queue age, indexing lag, retrieval failures, tool failures, and token usage.
- Run separate development, staging, and production environments with least-privilege service credentials.

## Operational states

- `local_only`: local indexing completed; no remote store configured.
- `pending_sync`: local indexing completed; durable remote sync work is outstanding.
- `synced`: manifest and all chunks are committed remotely at the current version.
- `sync_failed`: retries exhausted or a non-retryable error occurred; alert and replay from the outbox.
- `deleted`: deletion tombstone committed; derived data is eligible for cleanup.

The current `_supabase_synced` flag is a compatibility indicator, not a substitute for these durable states.
