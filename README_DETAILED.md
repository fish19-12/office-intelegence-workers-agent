The lazy ASGI wrapper's `/health/ready` check verifies that a shared secret and origin setting are present and that `LLM_PROVIDER` is not `mock`. It does not verify provider credentials or connectivity, and its check does not reject a wildcard origin.

# Office Workers Agent - Detailed Architecture & File Reference

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & System Design](#architecture--system-design)
3. [Core Components & File Descriptions](#core-components--file-descriptions)
4. [LLM Integration](#llm-integration)
5. [Advanced RAG System](#advanced-rag-system)
6. [Memory Management](#memory-management)
7. [Planning & Execution](#planning--execution)
8. [Multi-Agent System](#multi-agent-system)
9. [Tool Integration](#tool-integration)
10. [FastAPI Backend](#fastapi-backend)
11. [Installation & Setup](#installation--setup)
12. [Environment Variables](#environment-variables)
13. [Usage Examples](#usage-examples)

---

## Project Overview

### Current Source Snapshot (2026-10-02)

This section was checked against the current Python source, tests, `.env.example`, and `render.yaml`. It supersedes older examples below wherever they disagree. Older architecture prose is explanatory, not a guarantee that every advertised provider, tool, or workflow has been verified end to end.

#### Entrypoints and runtime

| File                             | Current responsibility                                                                                                                   |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `office_intelligence/api.py`     | Render/ASGI entrypoint; exports `app` from `app.asgi_app`.                                                                               |
| `app.py`                         | Lazy ASGI app and WSGI compatibility wrapper; imports `backend_api.app` on the first non-health request.                                 |
| `office_intelligence/runtime.py` | Shared upload path and lazily created `AgentOrchestrator` and LangChain executor.                                                        |
| `backend_api.py`                 | FastAPI routes, service-token validation, file handling, specialist-agent routing, planning, reports, and status.                        |
| `start.py`                       | Creates/reuses `.venv`, installs requirements, and prints integration instructions. Running it without arguments does not start the API. |
| `render.yaml`                    | Python 3.11.9 Render service, Uvicorn command, `/health/live`, and configured secret names.                                              |

For local development, use `python -m uvicorn office_intelligence.api:app --reload --host 127.0.0.1 --port 8000`. Render uses `uvicorn office_intelligence.api:app --host 0.0.0.0 --port $PORT`.

#### Current HTTP contract

Health endpoints `GET /health`, `GET /health/live`, and `GET /health/ready` do not require a service token. Other API requests require `Authorization: Bearer <token>`; `backend_api.py` validates an HS256 token using `OFFICE_INTELLIGENCE_SHARED_SECRET`, issuer `denbegaye-nextjs`, and audience `office-intelligence`.

| Route(s)                                                                    | Purpose                                                           |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `POST /query`, `POST /agent/run`                                            | RAG query and registered specialist-agent execution.              |
| `POST /upload`, `POST /upload-file`, `GET /download-file`, `GET /documents` | Document/file upload, retrieval, listing, and document ingestion. |
| `POST /plan`, `POST /execute-plan`, `GET /execute-plan/stream`              | Plan preview, confirmed execution, and streamed execution.        |
| `POST /report`, `/report/execute-block`, `/report/finalize`                 | Report generation and report-block/finalization operations.       |
| `POST /tools/credentials`, `GET /status`                                    | Tool credential update and runtime status.                        |

`POST /upload` is registered twice in `backend_api.py` with different request models: one handler expects a `files` list and ingests documents; a later handler accepts a single `file`. Resolve this collision before relying on either contract. The configured default upload limit is 50 MiB (`MAX_UPLOAD_BYTES`).

#### File-by-file source map

| Area                            | Files and responsibility                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Request routing and access      | `backend_api.py` handles the REST surface and signed-service-token middleware; `backend_agent_registry.py` maps public specialist IDs; `execution_contract.py` defines execution status/mode/artifact contracts.                                                                                                                                                       |
| Orchestration and planning      | `agent_orchestrator.py` composes document, retrieval, memory, planner, tools, and reports; `reactive_planner.py` creates and runs plans; `supervisor_agent.py` coordinates specialist agents; `agent_message_bus.py` passes inter-agent tasks.                                                                                                                         |
| Retrieval and embeddings        | `embeddings_rag.py` implements retrieval; `context_retriever.py` normalizes retrieval context; `embedding_service.py` provides embedding operations; `embedding_cache.py`, `embedding_jobs.py`, and `embedding_observability.py` support caching, batch jobs, and telemetry; `vector_store.py` and `supabase_client.py` support vector persistence.                    |
| Documents and reports           | `document_manager.py` manages ingestion; `ingestion.py` parses file formats; `chunker.py` splits text; `report_builder.py` builds report sections and exports.                                                                                                                                                                                                         |
| Memory and tools                | `memory_manager.py` stores/retrieves memory; `mcp_manager.py` validates and dispatches tool calls; `tools.py` implements tool credentials and external actions.                                                                                                                                                                                                        |
| Core specialist facade          | `specialist_agents.py` re-exports `DataAgent`, `ReportAgent`, `CommunicationAgent`, `RiskAgent`, and `SearchAgent`, implemented in `data_agent.py`, `report_agent.py`, `communication_agent.py`, `risk_agent.py`, and `search_agent.py`.                                                                                                                               |
| Data/document analyst modules   | `csv_analyst_agent.py`, `excel_analyst_agent.py`, `sql_analyst_agent.py`, `json_analyst_agent.py`, `financial_data_analyst.py`, `timeseries_forecaster_agent.py`, `multifile_correlation_analyzer.py`, `data_quality_analyzer.py`, and `ml_modeler_agent.py` handle their named data formats or analysis tasks.                                                        |
| Finance modules                 | `budget_actuals_analyzer.py`, `expense_auditor_agent.py`, `ar_aging_analyzer.py`, `cashflow_forecast_analyzer.py`, `payment_optimizer_agent.py`, `payroll_analyst.py`, `vendor_spend_analyzer.py`, and `invoice_processor_agent.py` implement finance-focused analyses.                                                                                                |
| Sales and customer modules      | `leads_analyzer_agent.py`, `campaign_performance_analyzer.py`, `sales_pipeline_analyst.py`, and `churn_analyzer_agent.py` cover leads, campaigns, pipeline, and churn.                                                                                                                                                                                                 |
| Operations and people modules   | `attendance_analyzer_agent.py`, `recruitment_analyst_agent.py`, `performance_review_analyzer.py`, `inventory_analyst_agent.py`, `supply_chain_analyzer.py`, `project_timeline_analyzer.py`, `survey_analyzer_agent.py`, and `sla_compliance_analyzer.py` cover their named operational domains.                                                                        |
| Documents, IT, and risk modules | `word_analyst_agent.py`, `ppt_analyst_agent.py`, `email_analyzer_agent.py`, `transcript_analyzer_agent.py`, `pdf_extractor_agent.py`, `image_processor_agent.py`, `log_analyst_agent.py`, `incident_analyzer_agent.py`, `access_rights_analyzer.py`, `license_tracker_analyzer.py`, and `risk_agent.py` handle their named document, IT, compliance, or risk analyses. |
| Tests                           | `tests/` contains tests for health, document processing, embeddings, execution contracts, LangChain, MCP tools, remote uploads, and tool credentials. These are focused tests, not full production acceptance coverage.                                                                                                                                                |

The registry in `backend_agent_registry.py` exposes many named analyst IDs beyond the five reusable core agents. `/agent/run` validates IDs against that registry; a registry entry alone is not evidence that every data shape/provider combination has been tested.

#### Provider and operational caveats

- `LLMFactory` selects OpenAI, DeepSeek, Hugging Face, Gemini, or Mock. `GenericHTTPAdapter` exists for direct use; the factory does not select it with a `generic` name. Do not document Anthropic as a working factory provider based only on installed packages.
- Missing provider configuration can result in Mock behavior. The lazy `/health/ready` wrapper checks that a shared secret and origin setting are present and `LLM_PROVIDER` is not `mock`; it does not validate credentials/connectivity or reject wildcard origins.
- `PythonExecutionTool` launches a local Python subprocess. It is not an OS/container sandbox; do not expose it to untrusted users or prompts.
- `ToolCredentialStore` writes credentials to process-local `tool_credentials.json` without encryption or tenant separation. Protect or replace it before production use.
- The runtime returns a shared, lazy `AgentOrchestrator` singleton. Verify that document, memory, vector, upload, and credential data are isolated by tenant before multi-customer deployment; request token claims alone do not establish data partitioning.
- Review `GET /execute-plan/stream`: the current implementation executes planned tool steps directly and does not accept the explicit `confirm` field used by `POST /execute-plan`.
- `requirements.txt` does not include every development test tool. Install test dependencies separately where needed and run the current suite before release.

Office Intelligence is a Python/FastAPI service for document retrieval and analysis, specialist data workflows, planning, and report generation. Microfinance is one supported domain; the current registry also includes finance, sales, operations, people, document, and IT analysis agents. This description is not an enterprise-readiness or compliance certification. The service combines:

- **Advanced Retrieval-Augmented Generation (RAG)** with hybrid search, re-ranking, and query decomposition
- **LLM-powered Planning** via a DAG-based reactive planner supporting parallel execution
- **Long-term Memory** with ChromaDB for persistent episodic, semantic, and procedural knowledge
- **Multi-Agent Architecture** with a supervisor coordinating five specialist agents
- **Tool Integration** including email, Google Workspace, and microfinance-specific utilities
- **FastAPI REST Backend** with document ingestion, semantic search, and report generation

### Key Capabilities

✅ Multi-format document ingestion (PDF, Excel, Word, images with OCR)  
✅ LLM-agnostic adapter layer (OpenAI, Deepseek, HuggingFace, Gemini, generic HTTP)  
✅ Hybrid search combining BM25 keyword matching + dense vector similarity  
✅ Cross-encoder re-ranking for relevance scoring  
✅ Query decomposition for complex multi-part questions  
✅ Contextual compression to reduce token consumption  
✅ Simple knowledge graph for entity relationship traversal  
✅ Self-RAG decisioning to determine when search is needed  
✅ Async DAG-based planning with dynamic replanning  
✅ Tool cooldown enforcement to prevent abuse  
✅ Credential management with optional/required validation

---

## Architecture & System Design

### High-Level Dataflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
├─────────────────────────────────────────────────────────────────┤
│                    AgentOrchestrator                             │
│  (Wires all components: RAG, Memory, Planner, Agents)          │
├─────────┬─────────────┬──────────────┬────────────┬───────────┤
│   RAG   │   Memory    │   Planner    │   Agents   │   Tools   │
├─────────┴─────────────┴──────────────┴────────────┴───────────┤
│                                                                 │
│  AdvancedRAG + ContextRetriever                                │
│  ├─ Hybrid Search (BM25 + Vector)                             │
│  ├─ Cross-Encoder Re-ranking                                  │
│  ├─ Query Decomposition (LLM)                                 │
│  ├─ Contextual Compression                                    │
│  ├─ Knowledge Graph Traversal                                 │
│  └─ Self-RAG Decisioning                                      │
│                                                                 │
│  MemoryManager (ChromaDB)                                      │
│  ├─ Episodic Memory (events + tool calls)                     │
│  ├─ Semantic Memory (facts + domain knowledge)                │
│  └─ Procedural Memory (workflows + how-tos)                   │
│                                                                 │
│  ReactivePlanner (DAG Execution)                               │
│  ├─ Dependency Graphs                                          │
│  ├─ Parallel Execution (asyncio)                              │
│  ├─ Conditional Branching                                      │
│  ├─ Dynamic Replanning                                         │
│  └─ Timeout/Retry Logic                                        │
│                                                                 │
│  Multi-Agent System                                            │
│  ├─ SupervisorAgent (goal decomposition, task assignment)     │
│  ├─ DataAgent (spreadsheet operations, aggregation)           │
│  ├─ ReportAgent (document generation)                         │
│  ├─ CommunicationAgent (email, notifications)                 │
│  ├─ RiskAgent (loan scoring, compliance)                      │
│  └─ SearchAgent (semantic retrieval)                          │
│                                                                 │
│  MCPManager (Tool Registry & Execution)                        │
│  ├─ Tool Schema Validation                                     │
│  ├─ Cooldown Enforcement                                       │
│  ├─ Audit Logging                                              │
│  └─ Error Wrapping                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Integration

1. **Document Ingestion** → DocumentManager parses and chunks files
2. **Embedding & Indexing** → AdvancedRAG builds hybrid indexes (BM25 + FAISS)
3. **Query Processing** → ContextRetriever orchestrates retrieval + LLM-driven enhancements
4. **Planning** → ReactivePlanner builds DAG from user goal
5. **Execution** → MCPManager executes tools with validation + cooldown
6. **Memory Recording** → Results stored in MemoryManager (episodic/semantic/procedural)
7. **Report Generation** → ReportBuilder creates structured outputs with code blocks

---

## Core Components & File Descriptions

### 1. **agent_orchestrator.py** — Main Orchestration Hub

**Purpose**: Wires all services (RAG, memory, planning, agents) into a unified interface.

**Key Classes**:

- `AgentOrchestrator`: Central coordinator managing document ingestion, querying, planning, and execution
- `QueryResult`: Result of semantic search
- `PlanPreview`: Dry-run preview of planned steps before execution
- `ConfirmationRequiredError`: Raised when plan contains irreversible tools

**Key Methods**:

- `ingest_file(file_path, metadata)` → Parses document, chunks, embeds, returns DocumentAsset
- `query(prompt, top_k, filters)` → Semantic search + optional Supabase hybrid retrieval
- `plan(goal, top_k)` → LLM generates DAG plan, validates tools, returns PlanPreview
- `execute_plan(plan, confirmed)` → Executes DAG asynchronously, records results in memory
- `generate_report(goal, query, top_k)` → Creates structured Word report with charts + tables

**Configuration**:

```python
config = {
    "llm_provider": "deepseek",  # or "openai", "huggingface", "gemini"
    "model": "gpt-4o-mini",
    "SUPABASE_URL": "https://...",  # optional
    "SUPABASE_KEY": "...",  # optional
}
agent = AgentOrchestrator(config=config)
```

---

### 2. **embeddings_rag.py** — Advanced Retrieval-Augmented Generation

**Purpose**: Core RAG engine combining hybrid search, re-ranking, decomposition, compression, and self-RAG.

**Key Classes**:

- `HuggingFaceEmbeddings`: Wrapper around SentenceTransformer for document/query encoding
- `AdvancedRAG`: Main RAG class orchestrating all retrieval features

**Features**:

#### a) Hybrid Search

Combines BM25 (keyword) + Vector (semantic) similarity:

- **BM25**: Tokenizes query, computes term frequency scores
- **Vector**: Encodes query via SentenceTransformer, searches via FAISS or numpy dot-product
- **Merging**: Deduplicates by chunk_id, takes max score per chunk
- **Fallback**: If FAISS unavailable, uses numpy dot-product

```python
rag = AdvancedRAG(llm=llm_instance)
rag.add_documents([
    ("The loan officer manages client ABC", {"officer": "John", "client_id": "ABC123"}),
    ("Interest rate is 18% per annum", {"policy": "rates"}),
])
results = rag.hybrid_search("What is John's interest rate?", top_k=5)
# Returns: [{'text': '...', 'meta': {...}, 'score': 0.92, 'source': 'bm25+vector'}, ...]
```

#### b) Cross-Encoder Re-ranking

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score candidate relevance:

```python
candidates = rag.hybrid_search(query, top_k=20)
reranked = rag.rerank(query, candidates, top_k=5)
# Returns top 5 by relevance score
```

#### c) Query Decomposition

Breaks complex questions into focused sub-queries via LLM:

```python
query = "Which officers manage the highest-risk clients in the eastern region?"
sub_queries = rag.decompose_query(query, llm=llm_instance)
# Returns: ["Who are the high-risk clients?", "Which officers manage them?", ...]
```

#### d) Contextual Compression

Extracts relevant sentences from chunks to reduce context size:

```python
full_text = "The loan officer John Smith manages client ABC Corp..."
compressed = rag.compress_context(full_text, "John's clients", llm=llm_instance)
# Returns: "The loan officer John Smith manages client ABC Corp"
```

#### e) Knowledge Graph Traversal

Simple entity relationship graph extraction from metadata:

```python
# Metadata keys: loan_officer, client_id, loan_id
# Edges: officer → manages → client → has → loan
kg_results = rag.traverse_kg("John Smith", depth=2)
# Returns related entities up to 2 hops away
```

#### f) Self-RAG Decisioning

LLM decides if search is needed or answer can be direct:

```python
should_search = rag.should_search("What is 2+2?", llm=llm_instance)
# Returns: False (simple math, no search needed)
```

**Key Methods**:

- `add_documents(text_meta_pairs)` → Indexes documents in BM25 + FAISS + memory
- `hybrid_search(query, top_k, metadata_filter)` → Combines BM25 + vector search
- `rerank(query, candidates, top_k)` → Cross-encoder re-ranking
- `decompose_query(query, llm)` → LLM-driven query splitting
- `compress_context(text, query, llm)` → Sentence extraction
- `traverse_kg(start, depth)` → Entity relationship exploration
- `should_search(query, llm)` → Self-RAG decision
- `retrieve(query, top_k, use_self_rag)` → High-level entry point orchestrating all steps

**Configuration**:

```python
EMBEDDING_MODEL = "BAAI/bge-m3"  # 1024-dim multi-lingual model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 1024
COMPACTION_THRESHOLD = 500  # Rebuild indexes after N inserts
```

---

### 3. **context_retriever.py** — Semantic Search Orchestrator

**Purpose**: High-level search interface combining local RAG + optional remote Supabase.

**Key Classes**:

- `RetrievedChunk`: Data class representing a retrieved document chunk
- `ContextRetriever`: Orchestrator for local + remote retrieval

**Key Methods**:

- `retrieve(query, top_k, filters)` → Calls AdvancedRAG.retrieve(), normalizes to RetrievedChunk
- `retrieve_for_goal(goal, top_k)` → Infers category from goal, retrieves relevant context
- `format_for_prompt(chunks, max_chars)` → Formats retrieved chunks for LLM injection
- `_check_sync_status()` → Passive tracking of Supabase sync state

**Integration**: Acts as adapter layer between AgentOrchestrator and AdvancedRAG, handling:

- Output normalization (AdvancedRAG returns dict, normalize to RetrievedChunk)
- Fallback to legacy `.query()` method if `.retrieve()` unavailable
- Supabase integration for remote persistence
- Backward compatibility with old systems

---

### 4. **memory_manager.py** — Long-Term Memory System

**Purpose**: Persistent memory using ChromaDB for episodic, semantic, and procedural knowledge.

**Key Classes**:

- `MemoryRecord`: Data class for memory entries
- `MemoryManager`: ChromaDB-backed memory store with compression

**Memory Types**:

1. **Episodic**: Events and tool execution history
   - "Sent payment reminder to CUST001 via email"
   - "Generated loan report for Q4 2024"
   - Indexed by timestamp and entity_ids

2. **Semantic**: Domain facts and policies
   - "Interest rate policy is 18% annually"
   - "Loan approval threshold is $100,000"
   - Indexed by topic/policy category

3. **Procedural**: Workflows and how-tos
   - "Steps to onboard new client: 1) KYC, 2) Risk assessment, 3) Approval"
   - Indexed by workflow name

**Key Methods**:

- `record(event_type, summary, detail, entity_ids, goal_tag)` → Log event to episodic memory
- `add_semantic_fact(text, detail)` → Store domain knowledge
- `add_procedural_note(workflow_name, steps, detail)` → Store process documentation
- `retrieve(query, top_k, memory_type)` → Semantic similarity search
- `get_prompt_context(query, top_k)` → Retrieve context for LLM prompt injection
- `_compress_short_term()` → LLM summarizes short-term events into long-term facts

**Storage**:

```python
# ChromaDB collections
- episodic_memory: {"summary": "...", "entity_ids": [...], "timestamp": "..."}
- semantic_memory: {"fact": "...", "category": "..."}
- procedural_memory: {"workflow": "...", "steps": [...]}
```

**Example Usage**:

```python
mem = MemoryManager(llm=llm_instance, persist_directory="memory_store")
mem.record("tool_executed", "Sent reminder email",
           detail={"tool": "email_send", "recipient": "client@example.com"},
           entity_ids=["CUST001"])
context = mem.get_prompt_context("reminder emails to clients", top_k=5)
```

---

### 5. **reactive_planner.py** — DAG-Based Reactive Planning

**Purpose**: LLM-powered planning with dependency graphs, parallel execution, and dynamic replanning.

**Key Classes**:

- `PlanStep`: Individual step in DAG with dependencies, conditions, timeouts, retries
- `PlanResult`: Complete plan execution result
- `ReactivePlanner`: DAG generator and executor

**Key Features**:

#### a) DAG Generation

LLM generates plan as directed acyclic graph:

```python
goal = "Send quarterly reports to all clients and log completion"
plan = planner.plan(goal)
# Returns:
# Step 1: Retrieve clients (no deps)
# Step 2: Generate report for each client (depends on Step 1)
# Step 3: Send emails (depends on Step 2)
# Step 4: Log to database (depends on Step 3)
```

#### b) Dependency Resolution

Topological sort identifies executable steps:

```python
step_2 = plan.steps[1]  # Send emails
print(step_2.depends_on)  # [0] (depends on step 1)
```

#### c) Parallel Execution

Asyncio concurrency for independent steps:

```python
# Steps 2a, 2b, 2c all execute in parallel (no deps on each other)
# Step 3 waits for all of Step 2 to complete
```

#### d) Conditional Branching

Steps execute only if conditions met:

```python
step = PlanStep(
    name="Archive old records",
    condition="total_records > 1000",  # Safe AST evaluation
    depends_on=[0]
)
```

#### e) Dynamic Replanning

On tool failure, LLM revises plan:

```python
# Step 2 fails: "Google Sheets API quota exceeded"
# Planner generates alternative: "Save to Excel instead"
# New step added, execution continues
```

#### f) Timeout & Retry Logic

```python
step = PlanStep(
    name="Long-running report",
    timeout_seconds=300,
    retry_count=2
)
```

**Key Methods**:

- `plan(goal, top_k)` → LLM generates initial DAG
- `execute(plan, confirmed)` → Async execution with replanning
- `_execute_step(step)` → Single step execution via MCPManager
- `_replan_after_failure(plan, failed_step, error)` → Generates alternative steps
- `_visualize_dag(plan)` → ASCII diagram of step dependencies

**Example**:

```python
planner = ReactivePlanner(llm, mcp_manager, memory, retriever)
goal = "Create and distribute monthly risk reports"
plan = planner.plan(goal, top_k=5)  # Retrieve context, generate DAG
result = planner.execute(plan, confirmed=True)  # Run async execution
print(f"Goal achieved: {result.goal_achieved}")
print(f"Duration: {result.total_duration_ms}ms")
```

---

### 6. **llm_interface.py** — LLM Abstraction Layer

**Purpose**: Unified interface for multiple LLM providers with adapters.

**Key Classes**:

- `BaseLLM`: Abstract base with `generate()` and `generate_json()` methods
- `MockLLM`: Stub for testing
- `OpenAIAdapter`: OpenAI API (gpt-3.5-turbo, gpt-4, etc.)
- `HuggingFaceAdapter`: HuggingFace Inference API
- `DeepseekAdapter`: Deepseek API (OpenAI-compatible)
- `GeminiAdapter`: Google Gemini
- `GenericHTTPAdapter`: Direct JSON HTTP adapter; it is not currently selected by `LLMFactory` with a `generic` provider name.
- `LLMFactory`: Factory for creating adapters

**Supported Models**:

`LLMFactory` provider names are `openai`/`chatgpt`, `deepseek`, `huggingface`/`hf`, `gemini`, and `mock`. Model availability depends on the selected adapter and credentials. The factory returns Mock behavior when some provider credentials are absent.

**Key Methods**:

- `generate(prompt, **kwargs)` → Synchronous text generation
- `generate_json(prompt, **kwargs)` → JSON parsing with fallbacks
- `generate_with_tools(prompt, tools, messages)` → Tool-calling interface
- `_augment_prompt_with_tools(prompt, tools)` → Format tools as schema
- `_parse_action_tool_call(text)` → Parse "ACTION: tool_name {...}" format

**Factory Usage**:

```python
from llm_interface import LLMFactory

# Option 1: From config dict
config = {
    "llm_provider": "deepseek",
    "api_key": "sk-...",
    "api_base": "https://api.deepseek.com/v1"
}
llm = LLMFactory.create(config)

# Option 2: With environment variables
llm = LLMFactory.create({"llm_provider": "openai"})  # Uses OPENAI_API_KEY env var

# Option 3: Mock for testing
llm = LLMFactory.create({"llm_provider": "mock"})
```

**JSON Generation**:

```python
response = llm.generate_json(
    "Generate a JSON object with fields: name, age, email"
)
# Attempts parsing:
# 1) Full JSON parse
# 2) Extract {...} substring
# 3) Extract [...] substring
# Raises ValueError if all fail
```

---

### 7. **document_manager.py** — Multi-Format Document Ingestion

**Purpose**: Parse and ingest documents from multiple formats with metadata enrichment.

**Supported Formats**:

- **PDF**: PyPDF for text extraction, fallback to PyMuPDF + Tesseract OCR
- **Excel**: pandas for .xlsx/.xls with per-sheet chunking
- **Word**: python-docx for .docx with paragraph-level metadata
- **CSV**: pandas with configurable delimiter
- **Images**: PIL + Tesseract OCR for .png/.jpg/.tiff
- **Images**: PIL/Tesseract processing where configured. The current API upload allow list is narrower; check `backend_api.py` before relying on image ingestion through HTTP.

**Key Classes**:

- `DocumentAsset`: Metadata about ingested document
- `DocumentManager`: Central ingestion orchestrator
- `DocumentIngestionError`: Parse error with context
- `UnsupportedFormatError`: Unsupported file type

**Key Methods**:

- `ingest(file_path, metadata)` → Parse, chunk, embed, return DocumentAsset
- `get_document(doc_id)` → Retrieve metadata + chunks
- `list_documents()` → All ingested documents with metadata
- `delete_document(doc_id)` → Remove from RAG + storage

**Chunking Strategy**:

```
1. Parse file into raw text
2. Detect language (langdetect)
3. Split by:
   - PDFs: Page boundaries
   - Excel: Sheet boundaries
   - Word: Paragraph boundaries
   - Images: Full page as one chunk
4. Apply RecursiveCharacterTextSplitter:
   - chunk_size=1024
   - overlap=200
5. Attach metadata (source, page, sheet, language)
```

**Example**:

```python
dm = DocumentManager(rag=advanced_rag_instance)
asset = dm.ingest(
    "loan_data.xlsx",
    metadata={
        "category": "Financial Reports",
        "priority": "High",
        "fiscal_year": 2024
    }
)
print(f"Ingested {asset.chunk_count} chunks from {asset.filename}")
```

---

### 8. **mcp_manager.py** — Tool Registry & Execution

**Purpose**: Register, validate, route, and execute tools with safety checks.

**Key Classes**:

- `ToolRegistry`: Maps tool names to schemas
- `MCPError`: Tool execution errors
- `MCPManager`: Main orchestrator

**Key Features**:

#### a) Tool Schema Validation

```python
schema = {
    "name": "send_email",
    "description": "Send an email",
    "inputSchema": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}
        },
        "required": ["to", "subject", "body"]
    }
}
mcp.register_tool("send_email", schema, EmailSendTool())
```

#### b) Credential Validation

```python
# Required credentials
credential_requirements = [
    {"env": "EMAIL_USERNAME", "required": True},
    {"env": "EMAIL_PASSWORD", "required": True}
]

# Optional credentials (any of)
credential_requirements = [
    {"env": ["SLACK_WEBHOOK", "TEAMS_WEBHOOK"], "any_of": True}
]
```

#### c) Cooldown Enforcement

Prevent tool abuse with rate limiting:

```python
mcp.call_tool("email_send", to="...", ...)
# Subsequent calls within cooldown period raise MCPError
```

#### d) Audit Logging

Track all tool calls:

```python
audit_log = mcp.get_audit_log()
# [
#   {
#     "tool": "email_send",
#     "args": {...},
#     "result": "success",
#     "timestamp": "2024-01-15T10:30:00Z",
#     "duration_ms": 250
#   }
# ]
```

**Key Methods**:

- `register_tool(name, schema, implementation)` → Add tool to registry
- `available_tools()` → List all registered tools
- `call_tool(name, **kwargs)` → Execute tool with validation
- `get_audit_log()` → Execution history
- `clear_audit_log(older_than)` → Cleanup old logs

---

### 9. **tools.py** — Microfinance & Integration Tools

**Purpose**: Implement specific tools for email, Google Workspace, and microfinance operations.

**Tool Classes**:

#### Communication Tools

- **EmailSendTool**: Send emails via SMTP (requires EMAIL_USERNAME, EMAIL_PASSWORD)
- **EmailInboxTool**: Read inbox via IMAP
- **SlackTool**: Send Slack messages (requires SLACK_WEBHOOK_URL)
- **TeamsTool**: Send Teams messages (requires TEAMS_WEBHOOK_URL)

#### Productivity Tools

- **GoogleDriveTool**: List/upload files to Google Drive
- **GoogleSheetsTool**: Read/write Google Sheets
- **GoogleCalendarTool**: List/create calendar events
- **PythonExecutionTool**: Launches Python in a subprocess/work directory; this is not a security sandbox.
- **PythonExecutionTool**: Launches Python in a subprocess/work directory; this is not a security sandbox.
- **VisualizationTool**: Create charts (matplotlib)
- **WordReportTool**: Generate .docx reports

#### Microfinance Tools

- **PaymentReminderTool**: Send payment reminders to clients
- **ClientOnboardingTool**: Onboard new clients (KYC, risk assessment)
- **RepaymentScheduleTool**: Generate repayment schedules
- **LoanReportDriveTool**: Generate and upload loan reports

**Key Base Class - ToolBase**:

```python
class ToolBase:
    args_schema: Dict[str, Any]  # JSON Schema for arguments
    credential_requirements: List[Dict]  # Required/optional credentials

    def execute(self, **kwargs) -> Any:
        """Tool implementation"""
        raise NotImplementedError

    def is_configured(self) -> bool:
        """Check if all required credentials present"""

    def missing_credentials(self) -> List[str]:
        """Return list of missing credential keys"""
```

**Example Tool Implementation**:

```python
class PaymentReminderTool(ToolBase):
    args_schema = {
        'type': 'object',
        'properties': {
            'client_id': {'type': 'string'},
            'message': {'type': 'string'},
            'channel': {'type': 'string', 'enum': ['email', 'sms']}
        },
        'required': ['client_id', 'message']
    }

    credential_requirements = [
        {"env": "EMAIL_USERNAME", "required": True},
        {"env": "EMAIL_PASSWORD", "required": True}
    ]

    def execute(self, client_id: str, message: str, channel: str = "email") -> Dict:
        # Send reminder via email or SMS
        if channel == "email":
            self._send_email(client_id, message)
        return {"status": "sent", "client_id": client_id}
```

---

### 10. **backend_api.py** — FastAPI REST Backend

**Purpose**: Expose orchestrator functionality as REST API endpoints.

**FastAPI Endpoints**:

#### a) **POST /query** — Semantic Search

Request:

```json
{
  "prompt": "Which clients have payment delays?",
  "top_k": 5,
  "filters": { "region": "Eastern" }
}
```

Response:

```json
{
  "answer": "The following clients...",
  "query": "Which clients have payment delays?",
  "chunks": [
    {"text": "...", "metadata": {...}, "source": "bm25+vector"}
  ]
}
```

#### b) **POST /upload** — Document Ingestion

Multipart form: `file`, `category` (optional metadata)
Response:

```json
{
  "ingested": [
    {
      "doc_id": "uuid-...",
      "filename": "report.pdf",
      "chunk_count": 42
    }
  ]
}
```

#### c) **GET /documents** — List Ingested Documents

Response:

```json
{
  "documents": [
    {
      "doc_id": "uuid-...",
      "filename": "report.pdf",
      "category": "Financial",
      "chunk_count": 42,
      "ingested_at": "2024-01-15T10:30:00Z",
      "metadata": {...}
    }
  ]
}
```

#### d) **POST /plan** — Generate Plan (Dry-Run)

Request:

```json
{
  "goal": "Generate and send quarterly reports",
  "top_k": 5
}
```

Response:

```json
{
  "goal": "Generate and send quarterly reports",
  "planned_tools": ["generate_report", "email_send", "log_event"],
  "irreversible_tools": ["email_send"],
  "needs_confirmation": true,
  "chunk_count": 5
}
```

#### e) **POST /execute-plan** — Execute Plan

#### e) **POST /execute-plan** — Execute Plan

Request:

```json
{
  "goal": "Generate and send quarterly reports",
  "confirm": true
}
```

Response:

```json
{
  "answer": "Successfully generated and sent 45 reports",
  "steps": [
    {
      "name": "Generate Report",
      "status": "success",
      "result": {...},
      "duration_ms": 2500
    }
  ],
  "goal_achieved": true,
  "total_duration_ms": 8750
}
```

#### f) **POST /report** — Generate Report

Request:

```json
{
  "goal": "Create Q4 2024 financial report",
  "query": "loan portfolio summary",
  "top_k": 10
}
```

Response:

```json
{
  "goal": "Create Q4 2024 financial report",
  "sections": [
    { "title": "Executive Summary", "content": "..." },
    { "title": "Loan Portfolio", "content": "..." }
  ],
  "html": "<h1>Q4 2024 Financial Report</h1>...",
  "ready_to_finalize": true
}
```

#### g) **GET /status** — System Status

Response:

```json
{
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "status": "healthy"
  },
  "rag": {
    "documents": 42,
    "chunks": 1250,
    "embedding_model": "BAAI/bge-m3"
  },
  "memory": {
    "episodic_records": 500,
    "semantic_facts": 200,
    "procedural_workflows": 15
  },
  "tools": ["email_send", "google_sheets", ...],
  "documents": ["report.pdf", "loan_data.xlsx", ...]
}
```

#### h) **POST /tools/credentials** — Update Tool Credentials

Request:

```json
{
  "credentials": {
    "EMAIL_USERNAME": "admin@company.com",
    "EMAIL_PASSWORD": "***"
  }
}
```

---

### 11. **specialist_agents.py** — Multi-Agent Specialists

**Purpose**: Task-specialized agents for data, reports, communication, risk, and search.

**Agent Classes**:

1. **DataAgent**: Spreadsheet operations, data aggregation
   - System Prompt: "Data specialist: read spreadsheets, compute aggregates, cleanse data."
   - Performs: summarize, filter, aggregate operations

2. **ReportAgent**: Document generation
   - System Prompt: "Report specialist: generate Word and PDF reports from structured data."
   - Performs: create sections, format, export

3. **CommunicationAgent**: Email & notifications
   - System Prompt: "Communication specialist: send emails and notifications."
   - Performs: route messages, send alerts

4. **RiskAgent**: Loan scoring & compliance
   - System Prompt: "Risk specialist: score loan applications and flag risky clients."
   - Performs: risk assessment, compliance checks

5. **SearchAgent**: Semantic retrieval
   - System Prompt: "Search specialist: RAG-based document search and retrieval."
   - Performs: retrieve relevant documents, rank results

**Base Class - BaseAgent**:

```python
class BaseAgent:
    SYSTEM_PROMPT: str  # Agent's role
    allowed_tools: List[str]  # Tools this agent can use

    def perform_task(self, payload: Dict[str, Any]) -> Any:
        """Execute specialized task"""
        raise NotImplementedError
```

---

### 12. **supervisor_agent.py** — Multi-Agent Orchestrator

**Purpose**: Decompose goals into tasks, assign to specialist agents, aggregate results.

**Key Method**:

- `supervise(goal, context)` → Decomposes goal, assigns tasks, aggregates results

**Process**:

1. **Decomposition**: LLM breaks goal into sub-tasks
2. **Assignment**: Match tasks to specialist agents
3. **Execution**: Concurrent execution via MessageBus
4. **Aggregation**: Combine results into final answer

**Example**:

```python
supervisor = SupervisorAgent(llm, mcp_manager, message_bus)
result = supervisor.supervise(
    goal="Generate risk assessment and send reports",
    context={"client_ids": ["C001", "C002"], "due_date": "2024-01-31"}
)
# Assigns:
#  - RiskAgent: Score risk for each client
#  - ReportAgent: Generate assessment documents
#  - CommunicationAgent: Send emails
```

---

### 13. **report_builder.py** — Structured Report Generation

**Purpose**: Generate multi-section reports with code blocks, tables, and Word export.

**Key Classes**:

- `CodeBlock`: Executable Python code with results
- `ReportSection`: Structured content with title + subsections
- `ReportResult`: Complete report with sections and HTML

**Key Methods**:

- `generate_report(goal, context, chunks)` → Multi-section LLM-powered report
- `add_section(title, content)` → Append section
- `add_code_block(title, code, language)` → Add executable snippet
- `to_html()` → Render as HTML
- `to_word(filename)` → Export .docx

---

### 14. **Supporting Files**

#### **ingestion.py** — File Reading Pipeline

- `DataIngestionPipeline`: Multi-format file parser
- Supports: PDF, Excel, Word, CSV, images
- Fallback strategies: OCR for scanned docs

#### **chunker.py** — Text Chunking

- `RecursiveCharacterTextSplitter`: LangChain-style splitter
- chunk_size=1024, overlap=200
- Preserves paragraph/sentence boundaries

#### **supabase_client.py** — Vector Database Integration

- Remote pgvector storage alongside local FAISS
- Hybrid retrieval: local FAISS + Supabase pgvector
- Async upsert for document embeddings

#### **agent_message_bus.py** — Inter-Agent Communication

- `MessageBus`: Pub-sub for agent tasks
- Supports concurrent agent execution

#### **base_agent.py** — Agent Base Class

- Common interface for all agents
- Task inbox, result collection

---

## LLM Integration

### Supported Providers

| Provider     | Model(s)                      | Config Key    | Auth               |
| ------------ | ----------------------------- | ------------- | ------------------ |
| OpenAI       | gpt-4, gpt-4o, gpt-3.5-turbo  | `openai`      | API_KEY            |
| Deepseek     | deepseek-chat, deepseek-coder | `deepseek`    | API_KEY + API_BASE |
| HuggingFace  | mistral-7b, llama-2, zephyr   | `huggingface` | HF_TOKEN           |
| Gemini       | gemini-pro, gemini-1.5        | `gemini`      | API_KEY            |
| Generic HTTP | Any OpenAI-compatible         | `generic`     | API_KEY + API_BASE |
| Mock         | N/A (testing)                 | `mock`        | None               |

### Configuration

```python
from llm_interface import LLMFactory

# Method 1: Config dictionary
config = {
    "llm_provider": "deepseek",
    "model": "deepseek-chat",
    "api_key": "sk-...",
    "api_base": "https://api.deepseek.com/v1",
    "temperature": 0.2,
    "max_tokens": 512
}
llm = LLMFactory.create(config)

# Method 2: Environment variables
os.environ["OPENAI_API_KEY"] = "sk-..."
llm = LLMFactory.create({"llm_provider": "openai"})

# Method 3: Custom HTTP endpoint
config = {
    "llm_provider": "generic",
    "api_base": "http://localhost:8000/v1",
    "api_key": "local-key"
}
llm = LLMFactory.create(config)
```

---

## Advanced RAG System

### Hybrid Search Algorithm

```
Input: query
├─ Tokenize query for BM25
├─ BM25 search
│  ├─ Compute term frequency scores
│  └─ Top K results by BM25 score
├─ Vector search
│  ├─ Encode query via SentenceTransformer
│  ├─ FAISS or numpy similarity search
│  └─ Top 3K results by similarity
├─ Merge results
│  ├─ Deduplicate by chunk_id
│  ├─ Take max score per chunk
│  └─ Sort by merged score
└─ Return top K candidates
```

### Query Decomposition

```
Input: "Which officers manage high-risk clients in the eastern region?"
├─ LLM generates sub-questions:
│  ├─ "Who are the high-risk clients?"
│  ├─ "Which officers manage them?"
│  └─ "Which clients are in the eastern region?"
├─ Search each sub-query independently
├─ Merge and deduplicate results
└─ Return consolidated results
```

### Compression Pipeline

```
Input: Full chunk text
├─ Split into sentences
├─ Encode each sentence
├─ Score by similarity to query
├─ Select top N relevant sentences
└─ Return compressed snippet
```

---

## Memory Management

### Storage Architecture

```
ChromaDB (File-based: memory_store/)
├─ episodic_memory
│  ├─ Records: {summary, entity_ids, timestamp, event_type}
│  └─ Vector: Encoded summary
├─ semantic_memory
│  ├─ Records: {fact, category, detail}
│  └─ Vector: Encoded fact
└─ procedural_memory
   ├─ Records: {workflow, steps, detail}
   └─ Vector: Encoded workflow
```

### Compression Strategy

**Short-term to Long-term**:

1. Collect recent episodic events (< 24h)
2. LLM summarizes: "3 payment reminders sent to clients in Eastern region"
3. Store summary in semantic memory
4. Delete individual events

---

## Planning & Execution

### DAG Execution Flow

```
┌─────────────┐
│ User Goal   │
└──────┬──────┘
       │ LLM generation
┌──────v──────────────────┐
│ DAG (Dependency Graph)  │
│ Step 1: Retrieve data   │
│ Step 2a: Generate report│
│ Step 2b: Compute stats  │
│ Step 3: Send email      │
└──────┬──────────────────┘
       │ Topological sort
┌──────v──────────────────┐
│ Executable steps        │
│ Round 1: [1]            │
│ Round 2: [2a, 2b]       │
│ Round 3: [3]            │
└──────┬──────────────────┘
       │ Async execution
┌──────v──────────────────┐
│ Tool calls via MCPManager│
│ Validate schemas        │
│ Check credentials       │
│ Apply cooldown          │
│ Execute tool            │
│ Record in memory        │
└──────┬──────────────────┘
       │ Replanning on failure
┌──────v──────────────────┐
│ Result aggregation      │
│ Goal achievement check  │
│ Timeline logging        │
└──────────────────────────┘
```

---

## Multi-Agent System

### Agent Coordination

```
┌────────────────┐
│ User Goal      │
└────────┬───────┘
         │
┌────────v────────────────────┐
│ SupervisorAgent             │
│  - Decompose goal           │
│  - Assign tasks             │
│  - Aggregate results        │
└────────┬────────────────────┘
         │
  ┌──────┴───────────────────────────┐
  │                                  │
┌─v──────────────┐    ┌──────────────v─┐
│ DataAgent      │    │ ReportAgent    │
│ (aggregates)   │    │ (generates)    │
└────────────────┘    └────────────────┘
  │                                  │
  └──────────────┬────────────────────┘
                 │
        ┌────────v────────┐
        │ ResultAggregation│
        └─────────────────┘
```

---

## Tool Integration

### Tool Lifecycle

```
1. Register
   ├─ Define schema (arguments + types)
   ├─ Define credentials (required/optional)
   └─ Implement execute() method

2. Discover
   ├─ Call mcp.available_tools()
   └─ Filter by allowed_tools for agent

3. Validate
   ├─ Check schema compliance
   ├─ Verify credentials present
   └─ Enforce cooldown period

4. Execute
   ├─ Call execute(**kwargs)
   ├─ Catch exceptions
   └─ Wrap in MCPError if failed

5. Record
   ├─ Store in audit log
   ├─ Record result in episodic memory
   └─ Track duration
```

---

## FastAPI Backend

### Middleware & CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File Upload Handling

```
POST /upload (multipart/form-data)
├─ Save to uploads/ directory
├─ Validate file format
├─ Ingest via DocumentManager
├─ Add to RAG index
└─ Return DocumentAsset metadata
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- Windows 10+ (or WSL)
- Tesseract OCR (for image documents)

### Quick Start (Windows)

```powershell
# 1. Bootstrap environment (creates .venv, installs dependencies)
python start.py

# 2. Verify installation
.\.venv\Scripts\python.exe -c "from agent_orchestrator import AgentOrchestrator; print('Installation successful')"

# 3. Copy environment template
copy .env.example .env

# 4. Fill in credentials in .env

# 5. Start backend API
.\.venv\Scripts\python.exe -m uvicorn office_intelligence.api:app --reload --host 127.0.0.1 --port 8000
```

### Manual Environment Setup

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract for OCR (Windows)
# Download installer: https://github.com/UB-Mannheim/tesseract/wiki
# Or via scoop:
scoop install tesseract
```

---

## Environment Variables

### Service Authentication and Readiness

```env
OFFICE_INTELLIGENCE_SHARED_SECRET=use-a-long-random-secret
OFFICE_INTELLIGENCE_ALLOWED_ORIGINS=http://localhost:3000
LLM_PROVIDER=deepseek
MAX_UPLOAD_BYTES=52428800
```

The trusted caller must sign a short-lived HS256 token with the same shared secret, issuer `denbegaye-nextjs`, and audience `office-intelligence`. Keep the secret server-side.

### LLM Configuration

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1  # Optional

# Deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# HuggingFace
HF_TOKEN=hf_...

# Gemini
GEMINI_API_KEY=AIza...

# Generic HTTP
LLM_API_URL=http://localhost:8000/v1
LLM_API_KEY=local-key
```

### Email Configuration

```bash
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=admin@company.com
EMAIL_PASSWORD=***
EMAIL_IMAP_SERVER=imap.gmail.com
```

### Google Integration

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_IMPERSONATED_USER=admin@company.com  # Optional
```

### Notifications

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
TEAMS_WEBHOOK_URL=https://outlook.webhook.office.com/webhookb2/.../
```

### Supabase (Optional)

```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIs...
```

### Document Processing

```bash
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
EMBEDDING_DIM=1024
```

---

## Usage Examples

### Example 1: Semantic Search

```python
from agent_orchestrator import AgentOrchestrator

config = {
    "llm_provider": "deepseek",
    "SUPABASE_URL": "https://xxx.supabase.co",
    "SUPABASE_KEY": "eyJhbGc..."
}
agent = AgentOrchestrator(config=config)

# Ingest document
asset = agent.ingest_file("loan_data.xlsx", {"category": "Financial"})
print(f"Ingested {asset.chunk_count} chunks")

# Semantic query
result = agent.query("Which clients have payment delays?", top_k=5)
print(result.answer)
for chunk in result.chunks:
    print(f"  - {chunk.text[:100]}...")
```

### Example 2: Planning & Execution

```python
# Generate plan (dry-run)
plan_preview = agent.plan(
    "Send payment reminders to overdue clients and log completion",
    top_k=5
)
print(f"Plan requires confirmation: {plan_preview.needs_confirmation}")
print(f"Tools: {plan_preview.planned_tools}")

# Execute plan (with confirmation if needed)
if plan_preview.needs_confirmation:
    print("Dangerous tools detected, requesting user confirmation...")
    confirmed = input("Proceed? (yes/no): ").lower() == "yes"
else:
    confirmed = True

if confirmed:
    result = agent.execute_plan(plan_preview, confirmed=True)
    print(f"Goal achieved: {result.goal_achieved}")
    for step in result.steps:
        print(f"  Step: {step['name']} ({step['status']})")
```

### Example 3: Report Generation

```python
# Generate multi-section report
report = agent.generate_report(
    goal="Create Q4 2024 risk assessment",
    query="high-risk clients, default rates, portfolio health",
    top_k=10
)

print(f"Report ready: {report.ready_to_finalize}")
for section in report.sections:
    print(f"\n{section.title}")
    print(section.content[:200] + "...")

# Export to Word
if report.ready_to_finalize:
    report.to_word("Q4_2024_Risk_Assessment.docx")
    print("✓ Report saved: Q4_2024_Risk_Assessment.docx")
```

### Example 4: Memory Usage

```python
from memory_manager import MemoryManager
from llm_interface import LLMFactory

llm = LLMFactory.create({"llm_provider": "deepseek"})
mem = MemoryManager(llm=llm, persist_directory="memory_store")

# Record event
mem.record(
    event_type="client_onboarded",
    summary="Client ABC Corp onboarded with KYC verification",
    detail={"kycid": "KYC-1234", "amount": 50000},
    entity_ids=["ABC123"],
    goal_tag="onboarding"
)

# Store fact
mem.add_semantic_fact(
    "Minimum loan amount is $5,000",
    detail={"policy": "lending_limits"}
)

# Retrieve context
context = mem.get_prompt_context(
    "What are the loan limits?",
    top_k=3
)
print(context)
```

### Example 5: Multi-Agent Coordination

```python
from supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent(
    llm=llm,
    mcp=mcp_manager,
    message_bus=message_bus
)

result = supervisor.supervise(
    goal="Generate risk scores for all clients and send reports",
    context={
        "client_ids": ["C001", "C002", "C003"],
        "deadline": "2024-01-31"
    }
)

print(f"Result: {result}")
```

---

## File Structure

```
microfinance-agent/
├── Core Orchestration
│   ├── agent_orchestrator.py      # Main coordination hub
│   ├── backend_api.py             # FastAPI REST endpoints
│   └── start.py                   # Bootstrap script
│
├── Retrieval & Search
│   ├── embeddings_rag.py          # Advanced RAG (hybrid, rerank, decompose)
│   ├── context_retriever.py       # High-level search orchestrator
│   ├── document_manager.py        # Multi-format document ingestion
│   └── ingestion.py               # File parsing pipeline
│
├── LLM Integration
│   └── llm_interface.py           # LLM adapters (OpenAI, Deepseek, etc.)
│
├── Memory & Planning
│   ├── memory_manager.py          # ChromaDB long-term memory
│   └── reactive_planner.py        # DAG-based planning & execution
│
├── Multi-Agent System
│   ├── base_agent.py              # Base agent class
│   ├── specialist_agents.py       # Data/Report/Comm/Risk/Search agents
│   ├── supervisor_agent.py        # Goal decomposition & coordination
│   └── agent_message_bus.py       # Inter-agent communication
│
├── Tool Integration
│   ├── tools.py                   # Email, Google, microfinance tools
│   └── mcp_manager.py             # Tool registry & execution
│
├── Report Generation
│   ├── report_builder.py          # Multi-section report generation
│   └── chunker.py                 # Text splitting
│
├── Cloud Integration
│   └── supabase_client.py         # Vector DB (Supabase pgvector)
│
├── Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example               # Environment template
│   ├── tool_credentials.json      # Tool credential store
│   ├── tool_credentials.json      # Created at runtime; local plaintext credential store
│   └── schema_v2.sql              # Supabase schema
│
├── Testing
│   ├── tests/
│   │   ├── test_health.py         # Integration tests
│   │   ├── test_mcp_manager.py    # Tool tests
│   │   └── test_tool_credentials.py
│   └── test_advanced_rag_demo.py  # RAG feature demo
│
├── Data
│   ├── uploads/                   # User file uploads
│   ├── memory_store/              # ChromaDB persistence
│   ├── data/                       # Sample datasets
│   └── tools/                      # Tool utilities
│
└── Documentation
    ├── README.md                  # Quick start
    ├── README_DETAILED.md         # This file
    ├── DETAILED_FUNCTIONALITY_REPORT_UPDATES.md
    └── implementations.md
```

---

## Advanced Features Summary

| Feature                      | Module               | Status                                               |
| ---------------------------- | -------------------- | ---------------------------------------------------- |
| **Hybrid Search**            | embeddings_rag.py    | ✅ Active                                            |
| **Cross-Encoder Re-ranking** | embeddings_rag.py    | ✅ Active                                            |
| **Query Decomposition**      | embeddings_rag.py    | ✅ Active                                            |
| **Contextual Compression**   | embeddings_rag.py    | ✅ Active                                            |
| **Knowledge Graph**          | embeddings_rag.py    | ✅ Active                                            |
| **Self-RAG**                 | embeddings_rag.py    | ✅ Active                                            |
| **ChromaDB Memory**          | memory_manager.py    | ✅ Active                                            |
| **DAG Planning**             | reactive_planner.py  | ✅ Active                                            |
| **Dynamic Replanning**       | reactive_planner.py  | ✅ Active                                            |
| **LLM Adapters**             | llm_interface.py     | Factory providers are listed in the current snapshot |
| **Multi-Agent System**       | specialist_agents.py | ✅ 5 agents                                          |
| **Tool Cooldown**            | mcp_manager.py       | ✅ Active                                            |
| **Audit Logging**            | mcp_manager.py       | ✅ Active                                            |
| **Supabase Integration**     | supabase_client.py   | ✅ Optional                                          |

---

## Deployment Notes

### Local Development

- Backend: `python -m uvicorn office_intelligence.api:app --reload --host 127.0.0.1 --port 8000` (use the project virtual environment)
- Logs: Check stdout for detailed execution traces
- Memory: Persists to `memory_store/` automatically

### Production Deployment

- The current Render blueprint uses `uvicorn office_intelligence.api:app --host 0.0.0.0 --port $PORT` and `/health/live`.
- Configure `OFFICE_INTELLIGENCE_SHARED_SECRET`, allowed origins, provider settings, and required integration credentials in the hosting platform.
- The current `ToolCredentialStore` is a local plaintext JSON file, not a secure vault. Replace it before production use.

---

**Last source review:** 2026-10-02
**Python version:** 3.11.9
