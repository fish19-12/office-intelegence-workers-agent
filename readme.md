# Office Worker Agent

A comprehensive enterprise AI system for automating microfinance workflows with advanced RAG, LLM-powered planning, multi-agent coordination, and tool integration.

**🎯 Key Features**:

- ✅ Advanced RAG (hybrid search, re-ranking, decomposition, compression, self-RAG)
- ✅ LLM-agnostic adapters (OpenAI, Deepseek, HuggingFace, Gemini, generic HTTP)
- ✅ DAG-based reactive planning with async execution
- ✅ ChromaDB long-term memory (episodic, semantic, procedural)
- ✅ Multi-agent system (Supervisor + 5 Specialists)
- ✅ Multi-format document ingestion (PDF, Excel, Word, images with OCR)
- ✅ Tool cooldown & credential management
- ✅ FastAPI REST backend with semantic search & report generation

📖 **[For Detailed Architecture & File Reference → See README_DETAILED.md](README_DETAILED.md)**

## Current Implementation Notes

The current API is implemented in `backend_api.py`; `office_intelligence/api.py` is the hosted ASGI entrypoint, and `app.py` provides lazy ASGI/legacy WSGI wrappers. Non-health API routes require an `Authorization: Bearer <signed-service-token>` header validated with `OFFICE_INTELLIGENCE_SHARED_SECRET`. The token uses HS256 and the configured issuer/audience; the health routes are exempt.

Current API families include `/query`, `/agent/run`, `/upload`, `/upload-file`, `/download-file`, `/documents`, `/plan`, `/execute-plan`, `/execute-plan/stream`, `/report`, `/report/execute-block`, `/report/finalize`, `/tools/credentials`, and `/status`. `/execute-plan` is hyphenated. `/health`, `/health/live`, and `/health/ready` are available without the service token.

Before exposing this service to customers, resolve the duplicate `POST /upload` registration in `backend_api.py`, isolate `PythonExecutionTool` in a real sandbox, protect `tool_credentials.json` (currently a process-local JSON credential store), and verify tenant isolation for agent memory, uploaded files, and credentials. This README does not certify production readiness.

---

## Quick Start (Windows)

### 1. Bootstrap the Python Environment

```powershell
python start.py
```

This creates `.venv` and installs `requirements.txt` when the environment is missing. It does not start the API server. The pinned runtime is Python 3.11.9 (`.python-version`).

### 2. Configure Credentials

Copy `.env.example` to `.env` and configure the shared service secret, allowed origins, an LLM provider/key, and only the integrations you intend to use. Keep secrets out of source control.

### 3. Start the Backend

```powershell
python -m uvicorn office_intelligence.api:app --reload --host 127.0.0.1 --port 8000
```

Backend is now at `http://localhost:8000`

### Deployment

Render uses `render.yaml` and starts the ASGI app with Uvicorn:

```text
uvicorn office_intelligence.api:app --host 0.0.0.0 --port $PORT
```

The configured entrypoint `office_intelligence.api:app` lazily imports the backend, so liveness checks do not construct the agent or embedding services. `/health/ready` reports whether the shared secret, non-wildcard origin configuration, and a non-mock LLM provider are configured.

### 4. Try It Out

Except for health checks, API requests require a short-lived signed service token from a trusted caller. Do not generate or expose this token in browser code. Use the web app's server-side proxy or an approved test-token tool.

---

## Core Components

| File                      | Purpose                                                  |
| ------------------------- | -------------------------------------------------------- |
| **agent_orchestrator.py** | Central hub wiring all services                          |
| **embeddings_rag.py**     | Advanced RAG: hybrid search, re-ranking, decomposition   |
| **llm_interface.py**      | LLM adapters (OpenAI, Deepseek, HuggingFace, Gemini)     |
| **memory_manager.py**     | ChromaDB long-term memory (episodic/semantic/procedural) |
| **reactive_planner.py**   | DAG-based planner with parallel execution                |
| **document_manager.py**   | Multi-format document ingestion                          |
| **mcp_manager.py**        | Tool registry, validation, cooldown                      |
| **tools.py**              | Email, Google, microfinance tools                        |
| **backend_api.py**        | FastAPI REST endpoints                                   |
| **specialist_agents.py**  | Data, Report, Communication, Risk, Search agents         |
| **supervisor_agent.py**   | Multi-agent orchestrator                                 |

📖 **[Full documentation → README_DETAILED.md](README_DETAILED.md)**

---

## REST API Endpoints

| Endpoint                                               | Method          | Purpose                                                                            |
| ------------------------------------------------------ | --------------- | ---------------------------------------------------------------------------------- |
| `/health`, `/health/live`, `/health/ready`             | GET             | Liveness and readiness checks; no service bearer token required.                   |
| `/query`                                               | POST            | Retrieve relevant document context and generate an answer.                         |
| `/agent/run`                                           | POST            | Run a registered specialist agent.                                                 |
| `/upload`                                              | POST            | Ingest documents; see the duplicate-route warning above.                           |
| `/upload-file`                                         | POST            | Store an uploaded file for later analysis.                                         |
| `/download-file`                                       | GET             | Download an uploaded file by path.                                                 |
| `/documents`, `/status`                                | GET             | List ingested documents and report service status.                                 |
| `/plan`, `/execute-plan`, `/execute-plan/stream`       | POST, POST, GET | Preview, execute, or stream a plan. Irreversible actions may require confirmation. |
| `/report`, `/report/execute-block`, `/report/finalize` | POST            | Generate, execute report blocks, and finalize reports.                             |
| `/tools/credentials`                                   | POST            | Update tool credentials; currently stored in the process-local JSON store.         |

All listed routes except health checks require a signed service token in the `Authorization: Bearer <token>` header.

---

## Environment Variables

The API also requires the following service settings for authenticated calls and a ready production health check:

```env
OFFICE_INTELLIGENCE_SHARED_SECRET=use-a-long-random-secret
OFFICE_INTELLIGENCE_ALLOWED_ORIGINS=http://localhost:3000
LLM_PROVIDER=deepseek
MAX_UPLOAD_BYTES=52428800
```

The same shared secret must be configured in the trusted server that signs requests. Do not expose it through a `NEXT_PUBLIC_` variable.

```bash
# LLM (choose one provider)
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
HF_TOKEN=hf_...

# Email
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=admin@company.com
EMAIL_PASSWORD=***

# Google (optional)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
TEAMS_WEBHOOK_URL=https://outlook.webhook.office.com/...

# Supabase (optional, for remote vector storage)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJhbGc...
```

See `.env.example` for all variables.

---

## Python API Examples

### Semantic Search

```python
from agent_orchestrator import AgentOrchestrator

agent = AgentOrchestrator(config={"llm_provider": "deepseek"})
agent.ingest_file("loan_data.xlsx")
result = agent.query("Which clients have payment delays?")
print(result.answer)
```

### Planning & Execution

```python
plan = agent.plan("Send payment reminders to overdue clients", top_k=5)
if plan.needs_confirmation:
    confirmed = input("Confirm? ").lower() == "yes"
else:
    confirmed = True

result = agent.execute_plan(plan, confirmed=confirmed)
print(f"Goal achieved: {result.goal_achieved}")
```

### Report Generation

```python
report = agent.generate_report(
    goal="Create Q4 2024 risk assessment",
    query="high-risk clients, default rates",
    top_k=10
)
if report.ready_to_finalize:
    report.to_word("Q4_Assessment.docx")
```

---

## LLM Adapters

`LLMFactory` currently selects OpenAI, DeepSeek, Hugging Face, Gemini, or Mock. `GenericHTTPAdapter` is available for direct use, but the factory does not currently select it by a `generic` provider name.

```python
from llm_interface import LLMFactory

# OpenAI
llm = LLMFactory.create({"llm_provider": "openai", "model": "gpt-4"})

# Deepseek (OpenAI-compatible)
llm = LLMFactory.create({"llm_provider": "deepseek"})

# HuggingFace
llm = LLMFactory.create({"llm_provider": "huggingface"})

# Gemini
llm = LLMFactory.create({"llm_provider": "gemini"})

# Generic (any OpenAI-compatible)
llm = LLMFactory.create({"llm_provider": "generic", "api_base": "..."})

# Mock (testing)
llm = LLMFactory.create({"llm_provider": "mock"})
```

---

## Advanced RAG Features

- **Hybrid Search**: BM25 (keyword) + Vector (semantic)
- **Re-ranking**: Cross-encoder for relevance
- **Query Decomposition**: LLM breaks complex questions
- **Compression**: Reduce token consumption
- **Knowledge Graph**: Entity relationships
- **Self-RAG**: Decide when search needed

---

## Multi-Agent System

Supervisor coordinates 5 specialist agents:

1. **DataAgent** - Spreadsheets, aggregation
2. **ReportAgent** - Document generation
3. **CommunicationAgent** - Email, notifications
4. **RiskAgent** - Loan scoring, compliance
5. **SearchAgent** - RAG-based retrieval

---

## Document Support

- **PDF**: Text + OCR for scanned
- **Excel**: .xlsx, .xls
- **Word**: .docx
- **CSV**: Configurable
- **Images**: .png, .jpg, and .jpeg; OCR depends on the selected ingestion path and installed Tesseract binary.

For OCR, install Tesseract:

```powershell
scoop install tesseract
```

---

## Web-App Integration

The related Next.js project calls this service through its server-side Office Intelligence proxy. Configure `OFFICE_INTELLIGENCE_URL` in the web app and use the same `OFFICE_INTELLIGENCE_SHARED_SECRET` on both sides. Keep the secret server-side. The web app repository has its own setup instructions.

---

## Architecture Overview

```
┌──────────────────────────────────┐
│    Frontend (Next.js)            │
├──────────────────────────────────┤
│    Backend REST API (FastAPI)    │
├──────────────────────────────────┤
│    AgentOrchestrator (Hub)       │
├────────┬────────┬────────┬───────┤
│  RAG   │ Memory │Planner │Agents │
├────────┴────────┴────────┴───────┤
│     Tools (Email, Google, etc.)  │
└──────────────────────────────────┘
```

---

## Documentation

- **[README_DETAILED.md](README_DETAILED.md)** - Current architecture, source map, API, and operational caveats
- **[DETAILED_FUNCTIONALITY_REPORT_UPDATES.md](DETAILED_FUNCTIONALITY_REPORT_UPDATES.md)** - Recent QA and follow-up notes
- **[ENTERPRISE_OPERATING_MODEL.md](ENTERPRISE_OPERATING_MODEL.md)** - Operational model and enterprise considerations

---

**Implementation status:** active codebase with automated tests, but production readiness depends on resolving the documented upload, code-execution, credential-storage, and tenant-isolation risks and validating the deployed services.

**Python version:** 3.11.9, as declared in `.python-version` and `render.yaml`.
