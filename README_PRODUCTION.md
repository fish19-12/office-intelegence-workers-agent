# Office Intelligence Workers Agent

Production-grade overview and system document for the Office Intelligence Workers Agent, a Python-based multi-agent enterprise intelligence platform for document ingestion, analysis, retrieval, planning, and automation across office workflows.

## 1. Overview

This project is designed to act as an intelligent operations layer for business teams working with structured and unstructured data. It combines:

- LLM-powered reasoning
- Retrieval-augmented generation (RAG)
- Multi-agent orchestration
- Document ingestion and parsing
- Analytics over CSV, Excel, SQL, JSON, and text sources
- Workspace-aware file handling
- API-first integrations for enterprise workflows

The system is built to support scenarios such as:

- analyzing financial, HR, sales, and operational documents
- answering business questions grounded in enterprise data
- orchestrating specialist agents for different domains
- managing plans, reports, and execution workflows
- integrating with external tools and document stores

At a high level, the platform behaves like an enterprise AI control plane: it accepts requests, retrieves relevant context, chooses specialist agents, executes tasks, and returns business-friendly results or generated artifacts.

---

## 2. Business and Technical Value

### Primary objectives

- Turn raw enterprise documents into searchable, analyzable context
- Support specialist AI agents for finance, risk, recruiting, sales, payroll, and other business functions
- Run a secure API layer with service authentication
- Provide reusable orchestration for long-running workflows and report generation
- Enable memory and retrieval for ongoing knowledge accumulation

### Primary users

- Operations teams
- Analysts
- Finance and risk teams
- Knowledge/workflow managers
- Internal AI application teams

---

## 3. Architecture Summary

The repository follows a layered enterprise architecture:

1. Interface and API layer
   - FastAPI app
   - health endpoints and protected endpoints
   - request validation and auth

2. Orchestration layer
   - supervisor and planner
   - execution contracts and multi-agent routing

3. Intelligence layer
   - LLM adapters
   - embedding generation
   - retrieval and ranking

4. Knowledge and memory layer
   - vector store integration
   - memory manager and document indexing

5. Domain analysis layer
   - specialist agents for different business domains
   - report, search, communication, risk, and data agents

6. Tooling and integration layer
   - external tool registry
   - MCP-inspired tool execution and credential management
   - optional connectors such as email, Google, Slack, Teams, and Supabase

7. Data ingestion and documents
   - CSV, Excel, Word, PDF, JSON, SQL-related assets
   - ingestion and chunking pipeline

This architecture is intentionally modular so that teams can extend domain logic without rewriting the core runtime.

---

## 4. Core Components

| Component | Purpose |
| --- | --- |
| `app.py` | ASGI/WSGI compatibility and lazy import wrapper |
| `backend_api.py` | Primary FastAPI backend and API contract |
| `office_intelligence/api.py` | ASGI application entrypoint used by deployment |
| `agent_orchestrator.py` | Central orchestration of services and requests |
| `supervisor_agent.py` | Coordinates multiple specialist agents |
| `specialist_agents.py` | Domain-level agent facade |
| `reactive_planner.py` | Creates and executes plans for tasks |
| `base_agent.py` | Shared base class for agent implementations |
| `llm_interface.py` | LLM abstraction layer |
| `embeddings_rag.py` | Retrieval and generation stack |
| `memory_manager.py` | Long-term and short-term memory operations |
| `document_manager.py` | File/document lifecycle management |
| `ingestion.py` | Document parsing and ingestion flow |
| `chunker.py` | Text segmentation for indexing and retrieval |
| `tools.py` | Tool registry and business integrations |
| `mcp_manager.py` | Tool validation and dispatch management |
| `vector_store.py` | Vector indexing / retrieval storage |
| `supabase_client.py` | Supabase-backed vector and storage integration |
| `report_builder.py` | Generated report assembly |

---

## 5. Multi-Agent Design

The system is organized around the idea that a single monolithic agent is inefficient for complex business work. Instead, the project creates specialist agents that each focus on a domain or task type.

Examples in the repository include:

- `attendance_analyzer_agent.py`
- `budget_actuals_analyzer.py`
- `cashflow_forecast_analyzer.py`
- `financial_data_analyst.py`
- `sales_pipeline_analyst.py`
- `invoice_processor_agent.py`
- `risk_agent.py`
- `search_agent.py`
- `report_agent.py`
- `communication_agent.py`
- `data_agent.py`

The general pattern is:

- a supervisor or orchestrator receives the task
- relevant context is retrieved
- an appropriate specialist agent is selected
- the task is executed with tool access and memory context
- results are returned or aggregated into a report

This pattern is well-suited to enterprise workflows because it allows domain-focused reasoning while keeping the central runtime reusable.

---

## 6. RAG, Retrieval, and Memory

The project includes a retrieval layer designed for enterprise knowledge grounding.

### Important modules

- `embeddings_rag.py` — retrieval, ranking, and context packaging
- `context_retriever.py` — normalization and retrieval context shaping
- `memory_manager.py` — persistent memory management
- `vector_store.py` — vector retrieval abstraction
- `embedding_service.py` — embedding generation service
- `embedding_cache.py` — reuse of embeddings for performance
- `embedding_jobs.py` and `embedding_observability.py` — background and telemetry support

### Why this matters

Enterprise AI systems fail when they answer from generic model memory without grounding. This project addresses that by:

- indexing uploaded documents and structured data
- retrieving semantically similar snippets
- reducing context to the most relevant declarations
- keeping memory for repeated workflows
- allowing report and task generation based on real evidence

---

## 7. Data and Document Ingestion

The project is designed for real business inputs, not just free-form prompts.

### Supported patterns

- CSV analysis with `csv_analyst_agent.py`
- Excel analysis with `excel_analyst_agent.py`
- JSON analysis with `json_analyst_agent.py`
- SQL-oriented workflows with `sql_analyst_agent.py`
- PDF extraction and parsing through `pdf_extractor_agent.py`
- file/document lifecycle management via `document_manager.py`

### Processing flow

1. File is uploaded or ingested
2. The document is parsed and normalized
3. Text is chunked for indexing
4. Embeddings are generated
5. Relevant chunks are retrieved during task execution
6. The agent reasons from fresh context and returns an answer or artifact

This makes the system broadly useful for operational and analytics work rather than only simple chat interactions.

---

## 8. API and Runtime Model

The REST API is implemented in `backend_api.py` and exposes a set of routes intended for services and internal tooling.

### Health and readiness

- `GET /health`
- `GET /health/live`
- `GET /health/ready`

These endpoints are intentionally lightweight and are designed for deployment health checks.

### Main operational endpoints

- `POST /query` — semantic or context-grounded query
- `POST /agent/run` — run a registered agent
- `POST /upload` — ingest documents
- `POST /upload-file` — store uploaded file for later analysis
- `GET /download-file` — retrieve uploaded file
- `GET /documents` — list ingested documents
- `POST /plan` — create a plan
- `POST /execute-plan` — execute a plan
- `GET /execute-plan/stream` — stream execution state
- `POST /report` — create report jobs
- `POST /report/execute-block` — execute report blocks
- `POST /report/finalize` — finalize report output
- `POST /tools/credentials` — update credentials
- `GET /status` — runtime status overview

### Authentication

The system is designed to require a signed service token for most routes. The expected claim is validated with a shared secret and configured issuer/audience settings.

The project also includes operational guidance warning that production hardening is still needed before public exposure. In particular, the README notes risks around duplicate endpoint registration, secure credential storage, tenant isolation, and sandboxing of Python execution tools.

---

## 9. Deployment Model

The deployment configuration is defined in `render.yaml`.

### Deployment stack

- Python 3.11.9 runtime
- Uvicorn ASGI server
- render-based hosting
- health checks on `/health/live`

### Typical command

```bash
uvicorn office_intelligence.api:app --host 0.0.0.0 --port $PORT
```

The project also includes a local startup helper in `start.py` that creates a `.venv` environment and installs dependencies when needed.

---

## 10. Environment and Configuration

The project relies on environment variables for LLM providers, secure tokens, file upload limits, and external services.

### Example variables

```env
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE=
GEMINI_API_KEY=
HF_TOKEN=

OFFICE_INTELLIGENCE_SHARED_SECRET=replace-with-the-same-long-random-server-secret
OFFICE_INTELLIGENCE_ALLOWED_ORIGINS=http://localhost:3000

SUPABASE_URL=
SUPABASE_KEY=

SLACK_WEBHOOK_URL=
TEAMS_WEBHOOK_URL=

EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_PASSWORD=
```

### Production requirements

- do not expose secrets in client-side code
- keep shared secrets server-side only
- configure explicit allowed origins, not wildcard origins
- validate credential storage and file permissions
- enforce tenant and workspace isolation
- isolate any Python execution sandboxing from the host machine

---

## 11. Tech Stack

### Core runtime

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic

### LLM stack

- LangChain
- LangChain OpenAI integrations
- provider-agnostic model adapters for OpenAI, DeepSeek, Hugging Face, Gemini, and generic APIs

### Analytics and data

- pandas
- matplotlib
- CSV/Excel/JSON parsing support
- document extraction pipelines

### Vector and retrieval

- embeddings
- hybrid and semantic retrieval patterns
- vector store abstraction

### Optional integrations

- Supabase
- Google APIs
- email/SMTP
- Slack/Teams webhooks

---

## 12. Repository Layout

```text
.
├── app.py
├── backend_api.py
├── base_agent.py
├── agent_orchestrator.py
├── supervisor_agent.py
├── specialist_agents.py
├── reactive_planner.py
├── llm_interface.py
├── embeddings_rag.py
├── memory_manager.py
├── vector_store.py
├── document_manager.py
├── ingestion.py
├── chunker.py
├── tools.py
├── mcp_manager.py
├── report_builder.py
├── office_intelligence/
│   ├── api.py
│   └── runtime.py
├── data/
├── uploads/
├── memory_store/
├── tests/
├── .env.example
├── render.yaml
├── requirements.txt
├── README.md
├── README_DETAILED.md
├── README_PRODUCTION.md
└── start.py
```

---

## 13. System Workflow

A typical request follows this lifecycle:

1. Client sends a request to the API
2. Request is validated, authenticated, and routed
3. The orchestrator determines the task context
4. Relevant documents or memory are retrieved
5. LLM reasoning is invoked through the configured provider
6. Specialist agents are selected for domain execution
7. Tools may be invoked for integrations or actions
8. Results are assembled into a response or report artifact
9. Optional long-running workflow states are persisted or streamed

This pattern makes the platform effective for both synchronous conversational analysis and asynchronous operational execution.

---

## 14. Security, Risk, and Production Readiness

This repository is a strong foundation for an enterprise AI workflow platform, but it should not be considered production-ready without additional hardening.

### Important cautions called out in project docs

- duplicate route registration in `backend_api.py` should be fixed
- Python execution should be sandboxed instead of running in the host environment
- tool credentials should be stored in a secure, isolated secret store
- tenant isolation must be verified for memory, uploads, and credentials
- wildcard origins and insecure secrets handling should be avoided
- API auth must be enforced consistently across all routes

These are not minor notes; they are critical deployment concerns for any multi-agent, tool-enabled service.

---

## 15. Testing and Validation

The repository includes various test files such as:

- `test_all_agents.py`
- `test_backend_integration.py`
- `test_integration.py`
- `test_sql_analyst_iterative.py`

The recommended validation approach is:

1. Verify environment variables are configured
2. Run the local Python environment bootstrap
3. Start the FastAPI app
4. Validate `/health` and `/health/ready`
5. Exercise API endpoints with service-token-authenticated calls
6. Confirm document ingestion and retrieval flows
7. Run targeted tests for agents and backend lifecycle

---

## 16. Recommended Production Hardening Checklist

Before production deployment, the team should address:

- service-to-service auth enforcement
- encrypted secret management
- secure storage for credentials and uploaded files
- tenant isolation and namespace separation
- strong input validation and file-type controls
- sandboxing of agent-executed code
- monitoring, traces, logs, and alerting
- rate limiting and abuse protections
- explicit backup and recovery strategy
- environment-specific configuration management

---

## 17. Suggested Development Roadmap

### Near term

- resolve route collisions and duplicated registrations
- centralize credential management
- confirm security posture for file upload and execution
- validate multi-agent orchestration end-to-end

### Mid term

- improve observability and structured logging
- add production-grade metrics and tracing
- harden vector and document ingestion pipelines
- formalize resource isolation for tenants and workspaces

### Long term

- create managed AI workflow orchestration for business units
- add governance, audit logs, and compliance features
- expand integrations and policy engines
- support enterprise deployment and monitoring standards

---

## 18. Final Assessment

This repository represents a serious and well-structured foundation for an enterprise office-intelligence platform. It combines document intelligence, multi-agent orchestration, retrieval, analytics, and API-driven orchestration into a single cohesive architecture.

It is especially strong in:

- modular agent design
- multi-domain business analysis support
- retrieval and document context grounding
- workflow orchestration and execution patterns
- flexible integration model for enterprise automation

It still requires production hardening before public or broad enterprise deployment, but the architecture and project structure are sufficiently rich to support a robust business AI system.

---

## 19. References

- `README.md` — high-level project overview
- `README_DETAILED.md` — deeper technical architecture and file map
- `render.yaml` — deployment configuration
- `.env.example` — runtime environment configuration
- `backend_api.py` — primary API surface
- `agent_orchestrator.py` — central orchestration engine

This document is intended to serve as the production-level system overview for the repository and should be used alongside the implementation documents for operational decisions and engineering planning.
