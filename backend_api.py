from __future__ import annotations

import json
import logging
import os
import base64
import hashlib
import hmac
import time
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request as URLRequest, urlopen

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from urllib.parse import quote_plus

from agent_orchestrator import ConfirmationRequiredError
from backend_agent_registry import is_known_agent
from csv_analyst_agent import run_csv_analyst
from excel_analyst_agent import run_excel_analyst
from sql_analyst_agent import run_sql_analyst
from financial_data_analyst import run_financial_analyst
from payroll_analyst import run_payroll_analyst
from sales_pipeline_analyst import run_sales_pipeline_analyst
from json_analyst_agent import run_json_analyst
from churn_analyzer_agent import run_churn_analyzer
from attendance_analyzer_agent import run_attendance_analyzer
from word_analyst_agent import run_word_analyst
from ppt_analyst_agent import run_ppt_analyst
from email_analyzer_agent import run_email_analyzer
from transcript_analyzer_agent import run_transcript_analyzer
from invoice_processor_agent import run_invoice_processor
from recruitment_analyst_agent import run_recruitment_analyst
from performance_review_analyzer import run_performance_review_analyzer
from budget_actuals_analyzer import run_budget_actuals_analyzer
from expense_auditor_agent import run_expense_auditor
from ar_aging_analyzer import run_ar_aging_analyzer
from cashflow_forecast_analyzer import run_cashflow_forecast_analyzer
from vendor_spend_analyzer import run_vendor_spend_analyzer
from payment_optimizer_agent import run_payment_optimizer
from project_timeline_analyzer import run_project_timeline_analyzer
from sla_compliance_analyzer import run_sla_compliance_analyzer
from inventory_analyst_agent import run_inventory_analyst
from supply_chain_analyzer import run_supply_chain_analyzer
from leads_analyzer_agent import run_leads_analyzer
from campaign_performance_analyzer import run_campaign_performance_analyzer
from survey_analyzer_agent import run_survey_analyzer
from access_rights_analyzer import run_access_rights_analyzer
from license_tracker_analyzer import run_license_tracker_analyzer
from incident_analyzer_agent import run_incident_analyzer
from ml_modeler_agent import run_ml_modeler
from log_analyst_agent import run_log_analyst
from image_processor_agent import run_image_processor
from pdf_extractor_agent import run_pdf_extractor
from timeseries_forecaster_agent import run_timeseries_forecaster
from multifile_correlation_analyzer import run_multifile_correlation_analyzer
from data_quality_analyzer import run_data_quality_analyzer
from langchain_agent import LangChainAgentExecutor
from report_builder import CodeBlock, ReportSection, ReportResult as BuiltReportResult
from fastapi.responses import StreamingResponse
from tools import ToolCredentialStore
from office_intelligence.runtime import UPLOAD_DIR, get_agent, get_langchain_executor
from execution_contract import ExecutionMode, ExecutionStatus, create_execution_record

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SERVICE_TOKEN_ISSUER = "denbegaye-nextjs"
SERVICE_TOKEN_AUDIENCE = "office-intelligence"


def _decode_base64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify_service_token(token: str) -> Dict[str, Any]:
    secret = os.environ.get("OFFICE_INTELLIGENCE_SHARED_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="Office Intelligence auth is not configured.")

    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid service token.")

    header_part, payload_part, signature_part = parts
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    expected_signature = hmac.new(
        secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    try:
        actual_signature = _decode_base64url(signature_part)
        header = json.loads(_decode_base64url(header_part))
        payload = json.loads(_decode_base64url(payload_part))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid service token.") from exc

    if not hmac.compare_digest(actual_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid service token.")
    if header.get("alg") != "HS256" or header.get("typ") != "JWT":
        raise HTTPException(status_code=401, detail="Invalid service token.")
    if payload.get("iss") != SERVICE_TOKEN_ISSUER or payload.get("aud") != SERVICE_TOKEN_AUDIENCE:
        raise HTTPException(status_code=401, detail="Invalid service token claims.")

    try:
        expires_at = int(payload["exp"])
        issued_at = int(payload["iat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid service token timestamps.") from exc

    now = int(time.time())
    if expires_at <= now or issued_at > now + 30:
        raise HTTPException(status_code=401, detail="Service token expired or not yet valid.")
    return payload

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_BUCKET = "office-intelligence-uploads"


def _max_upload_bytes() -> int:
    configured_limit = int(os.environ.get("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
    if configured_limit <= 0:
        raise RuntimeError("MAX_UPLOAD_BYTES must be greater than zero.")
    return configured_limit


def _download_supabase_file(file_url: str, file_name: str) -> Path:
    configured_supabase_url = os.environ.get("SUPABASE_URL")
    if not configured_supabase_url:
        raise HTTPException(status_code=503, detail="Supabase storage is not configured.")

    expected_url = urlsplit(configured_supabase_url)
    signed_url = urlsplit(file_url)
    signed_path_prefix = f"/storage/v1/object/sign/{UPLOAD_BUCKET}/"
    if (
        signed_url.scheme != expected_url.scheme
        or signed_url.netloc.lower() != expected_url.netloc.lower()
        or not signed_url.path.startswith(signed_path_prefix)
        or not parse_qs(signed_url.query).get("token")
    ):
        raise HTTPException(status_code=403, detail="Invalid signed storage URL.")

    safe_name = Path(file_name).name
    suffix = Path(safe_name).suffix.lower()
    max_bytes = _max_upload_bytes()
    temporary_path: Optional[Path] = None
    try:
        with urlopen(URLRequest(file_url, method="GET"), timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the configured upload limit of {max_bytes} bytes.",
                )

            with tempfile.NamedTemporaryFile(
                mode="wb", dir=UPLOAD_DIR.resolve(), suffix=suffix, prefix="analysis-", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                total_bytes = 0
                while chunk := response.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=413, detail="File exceeds the configured upload limit."
                        )
                    temporary_file.write(chunk)
                if total_bytes == 0:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return temporary_path
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        logger.warning("Failed to download signed Supabase object: %s", exc)
        raise HTTPException(status_code=502, detail="Could not download the uploaded file.") from exc


def serialize_datetime_objects(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime_objects(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_datetime_objects(item) for item in obj]
    return obj


class QueryRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, str]] = None


class QueryResponse(BaseModel):
    answer: str
    query: str
    chunks: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    ingested: List[Dict[str, Any]]


class PlanRequest(BaseModel):
    goal: str
    top_k: Optional[int] = 5


class PlanPreviewResponse(BaseModel):
    goal: str
    planned_tools: List[str]
    irreversible_tools: List[str]
    needs_confirmation: bool
    chunk_count: int


class ExecutePlanRequest(BaseModel):
    goal: str
    top_k: Optional[int] = 5
    confirm: bool = False


class PlanExecutionResponse(BaseModel):
    answer: str
    steps: List[Dict[str, Any]]
    partial: bool
    goal_achieved: Optional[bool]
    total_duration_ms: Optional[int]


class ReportRequest(BaseModel):
    goal: str
    query: Optional[str] = None
    top_k: Optional[int] = 5


class ReportResponse(BaseModel):
    goal: str
    sections: List[Dict[str, Any]]
    pending_code_blocks: List[Dict[str, Any]]
    html: Optional[str]
    ready_to_finalize: bool


class DocumentAssetResponse(BaseModel):
    doc_id: str
    filename: str
    category: str
    chunk_count: int
    ingested_at: str
    metadata: Dict[str, Any]
    _supabase_synced: bool


class StatusResponse(BaseModel):
    llm: Dict[str, Any]
    rag: Dict[str, Any]
    memory: Dict[str, Any]
    tools: Any
    documents: Dict[str, Any]


class ToolCredentialsRequest(BaseModel):
    credentials: Dict[str, str]


class AgentExecutionRequest(BaseModel):
    agent_id: str
    prompt: str
    execution_id: Optional[str] = None
    trace_id: Optional[str] = None
    mode: Optional[str] = "auto"
    use_langchain: bool = True
    table_csv: Optional[str] = None
    table_json: Optional[List[Dict[str, Any]]] = None
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    db_file: Optional[str] = None
    top_k: Optional[int] = 5
    confirm: bool = False


class AgentExecutionResponse(BaseModel):
    agent_id: str
    mode: str
    answer: str
    execution_id: Optional[str] = None
    trace_id: Optional[str] = None
    status: str = "queued"
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _resolve_uploaded_agent_input(
    file_path: Optional[str],
    table_csv: Optional[str],
    table_json: Optional[List[Dict[str, Any]]],
) -> tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
    """Convert uploaded tabular files into the inputs specialist agents consume."""
    if table_csv is not None or table_json is not None or not file_path:
        return table_csv, table_json, None

    requested_path = Path(file_path)
    resolved_path = requested_path if requested_path.is_absolute() else UPLOAD_DIR / requested_path
    resolved_path = resolved_path.resolve()
    upload_root = UPLOAD_DIR.resolve()
    if upload_root not in resolved_path.parents and resolved_path != upload_root:
        raise HTTPException(status_code=403, detail="Uploaded file path is outside the upload directory.")
    if not resolved_path.is_file():
        raise HTTPException(status_code=400, detail=f"Uploaded file not found: {file_path}")

    suffix = resolved_path.suffix.lower()
    if suffix in {".csv", ".txt", ".log", ".vtt", ".mbox"}:
        return resolved_path.read_text(encoding="utf-8-sig", errors="replace"), None, str(resolved_path)
    if suffix == ".json":
        try:
            value = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {file_path}") from exc
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return None, value, str(resolved_path)
        return None, [{"value": value}], str(resolved_path)
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd

            frame = pd.read_excel(resolved_path)
            return None, frame.where(frame.notna(), None).to_dict(orient="records"), str(resolved_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read spreadsheet: {file_path}") from exc

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(resolved_path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return None, [{
                "filename": resolved_path.name,
                "page_count": len(reader.pages),
                "text_content": text,
                "tables_found": 0,
                "file_size": resolved_path.stat().st_size,
            }], str(resolved_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read PDF: {file_path}") from exc

    if suffix in {".png", ".jpg", ".jpeg"}:
        try:
            from PIL import Image

            with Image.open(resolved_path) as image:
                return None, [{
                    "filename": resolved_path.name,
                    "file_size": resolved_path.stat().st_size,
                    "width": image.width,
                    "height": image.height,
                    "format": image.format or suffix.removeprefix("."),
                    "color_space": image.mode,
                }], str(resolved_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read image: {file_path}") from exc

    if suffix == ".pptx":
        try:
            from pptx import Presentation

            presentation = Presentation(str(resolved_path))
            rows = []
            for index, slide in enumerate(presentation.slides, start=1):
                text_content = "\n".join(
                    shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text
                )
                rows.append({
                    "presentation_name": resolved_path.name,
                    "slide_number": index,
                    "slide_count": len(presentation.slides),
                    "text_content": text_content,
                    "speaker_notes": "",
                })
            return None, rows, str(resolved_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to read PowerPoint file: {file_path}") from exc

    return None, None, str(resolved_path)


app = FastAPI(title="Microfinance Worker Agent API", version="1.0.0")


def _parse_allowed_origins(raw_value: str) -> List[str]:
    origins: List[str] = []
    for entry in (raw_value or "").split(","):
        origin = entry.strip().rstrip("/")
        if not origin or origin == "*":
            continue
        if not origin.startswith(("http://", "https://")):
            raise ValueError(f"Invalid allowed origin: {origin}")
        origins.append(origin)
    return origins


try:
    allowed_origins = _parse_allowed_origins(
        os.environ.get("OFFICE_INTELLIGENCE_ALLOWED_ORIGINS", "http://localhost:3000")
    )
except ValueError as exc:
    logger.warning("Invalid OFFICE_INTELLIGENCE_ALLOWED_ORIGINS configuration: %s", exc)
    allowed_origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    if request.url.scheme == "https":
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )
    return response


@app.middleware("http")
async def authenticate_service_request(request: Request, call_next):
    started_at = time.perf_counter()
    caller = "anonymous"

    if request.url.path not in {"/health", "/health/live", "/health/ready"}:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("office_request_denied method=%s path=%s reason=missing_bearer", request.method, request.url.path)
            return JSONResponse({"detail": "Authentication required."}, status_code=401)
        try:
            claims = _verify_service_token(auth_header[7:].strip())
            caller = str(claims.get("sub", "unknown"))
            request.state.authenticated_caller = caller
            request.state.tenant_id = str(claims.get("tenant_id", "default"))
            request.state.authenticated_claims = claims
        except HTTPException as exc:
            logger.warning("office_request_denied method=%s path=%s reason=%s", request.method, request.url.path, exc.detail)
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "office_request caller=%s method=%s path=%s status=%s duration_ms=%s",
            caller,
            request.method,
            request.url.path,
            getattr(locals().get("response"), "status_code", "error"),
            duration_ms,
        )

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health/live")
async def health_live() -> Dict[str, str]:
    return {"status": "ok", "service": "office-intelligence", "check": "liveness"}


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    """Report whether required production configuration is available."""
    checks = {
        "shared_secret": bool(os.environ.get("OFFICE_INTELLIGENCE_SHARED_SECRET")),
        "allowed_origins": bool(allowed_origins) and "*" not in allowed_origins,
        "llm_provider": os.environ.get("LLM_PROVIDER", "deepseek").lower() != "mock",
    }
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@app.post("/query", response_model=QueryResponse)
async def query_agent(payload: QueryRequest) -> QueryResponse:
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    result = get_agent().query(payload.prompt, top_k=payload.top_k, filters=payload.filters)
    return {
        "answer": result.answer,
        "query": result.query,
        "chunks": [chunk.to_dict() for chunk in result.chunks],
    }


@app.post("/agent/run", response_model=AgentExecutionResponse)
async def run_agent(request: Request, payload: AgentExecutionRequest) -> AgentExecutionResponse:
    if not payload.agent_id or not payload.agent_id.strip():
        raise HTTPException(status_code=400, detail="agent_id is required.")
    if not is_known_agent(payload.agent_id):
        raise HTTPException(status_code=400, detail=f"Unknown agent_id: {payload.agent_id}")
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required.")

    resolved_csv = payload.table_csv
    resolved_json = payload.table_json
    resolved_path = payload.file_path
    downloaded_file_path: Optional[Path] = None

    answer = ""
    execution_record = create_execution_record(
        tenant_id=getattr(request.state, "tenant_id", "default"),
        user_id=getattr(request.state, "authenticated_caller", "system"),
        agent_id=payload.agent_id,
        mode=ExecutionMode(payload.mode or "auto"),
        prompt=payload.prompt,
        attachments=(
            [resolved_path]
            if resolved_path
            else ([Path(payload.file_name).name] if payload.file_url and payload.file_name else [])
        ),
        parameters={
            "top_k": payload.top_k,
            "use_langchain": payload.use_langchain,
            "confirm": payload.confirm,
            "table_csv_present": bool(payload.table_csv),
            "table_json_present": bool(payload.table_json),
            "db_file": payload.db_file,
        },
        metadata={"source": "backend_api"},
    )
    if payload.execution_id:
        execution_record.execution_id = payload.execution_id
    if payload.trace_id:
        execution_record.trace_id = payload.trace_id
    metadata: Dict[str, Any] = {
        "agent_id": payload.agent_id,
        "mode": payload.mode,
    }
    if payload.file_url and payload.file_name:
        metadata["input_file"] = Path(payload.file_name).name
    elif resolved_path:
        metadata["input_file"] = Path(resolved_path).name

    try:
        if payload.file_url:
            downloaded_file_path = _download_supabase_file(payload.file_url, payload.file_name or "")
            payload = payload.model_copy(
                update={
                    "file_path": str(downloaded_file_path),
                    **(
                        {"db_file": str(downloaded_file_path)}
                        if payload.agent_id == "sql-analyst"
                        else {}
                    ),
                }
            )

        resolved_csv, resolved_json, resolved_path = _resolve_uploaded_agent_input(
            payload.file_path,
            payload.table_csv,
            payload.table_json,
        )
        payload = payload.model_copy(
            update={
                "table_csv": resolved_csv,
                "table_json": resolved_json,
                "file_path": resolved_path or payload.file_path,
            }
        )

        if payload.agent_id == "csv-analyst":
            if not payload.table_csv:
                raise HTTPException(status_code=400, detail="CSV data is required for csv-analyst.")
            csv_result = run_csv_analyst(payload.table_csv, payload.prompt)
            answer = csv_result.get("answer", "")
            metadata.update({
                "source": "csv_analyst",
                "chart_created": csv_result.get("chart_created", False),
                "chart_png_base64": csv_result.get("chart_png_base64"),
            })
        elif payload.agent_id == "sql-analyst":
            # sql-analyst supports table JSON or an uploaded DB file path.
            if payload.db_file:
                db_path = payload.db_file
                if not os.path.isabs(db_path):
                    db_path = str(UPLOAD_DIR / db_path)
                if not Path(db_path).exists():
                    raise HTTPException(status_code=400, detail=f"Database file not found: {payload.db_file}")
                sql_result = run_sql_analyst(db_path, None, None, payload.prompt)
                answer = sql_result.get("answer", json.dumps(sql_result))
                metadata.update({"source": "sql_analyst_db", "db_file": db_path})
            elif payload.table_json is not None:
                sql_result = run_sql_analyst(None, None, payload.table_json, payload.prompt)
                answer = sql_result.get("answer", json.dumps(sql_result))
                metadata["source"] = "sql_analyst_json"
            else:
                raise HTTPException(status_code=400, detail="sql-analyst requires table_json input or an uploaded DB file.")
        elif payload.agent_id == "excel-analyst":
            if payload.table_json is not None:
                excel_result = run_excel_analyst(None, payload.table_json, payload.prompt)
                answer = json.dumps(excel_result)
                metadata["source"] = "excel_analyst_json"
            else:
                raise HTTPException(status_code=400, detail="excel-analyst requires table_json input or an uploaded Excel file.")
        elif payload.agent_id == "financial-data":
            if payload.table_csv:
                fin_result = run_financial_analyst(payload.table_csv, None, payload.prompt)
                answer = fin_result.get("answer", json.dumps(fin_result))
                metadata.update({
                    "source": "financial_analyst_csv",
                    "rows": fin_result.get("rows"),
                    "columns": fin_result.get("columns"),
                    "chart_created": fin_result.get("chart_created", False),
                    "chart_png_base64": fin_result.get("chart_png_base64"),
                })
            else:
                raise HTTPException(status_code=400, detail="financial-data requires CSV input.")
        elif payload.agent_id == "payroll-analyst":
            if payload.table_csv:
                payroll_result = run_payroll_analyst(payload.table_csv, None, payload.prompt)
                answer = json.dumps(payroll_result)
                metadata["source"] = "payroll_analyst_csv"
            else:
                raise HTTPException(status_code=400, detail="payroll-analyst requires CSV input.")
        elif payload.agent_id == "sales-pipeline":
            if payload.table_csv:
                sales_result = run_sales_pipeline_analyst(payload.table_csv, None, payload.prompt)
                answer = json.dumps(sales_result)
                metadata["source"] = "sales_pipeline_analyst_csv"
            else:
                raise HTTPException(status_code=400, detail="sales-pipeline requires CSV input.")
        elif payload.agent_id == "json-analyst":
            # JSON analyst can accept either table_json or raw JSON in prompt
            if payload.table_json is not None:
                json_result = run_json_analyst(json.dumps(payload.table_json), payload.prompt)
                answer = json.dumps(json_result)
                metadata["source"] = "json_analyst_json"
            else:
                raise HTTPException(status_code=400, detail="json-analyst requires table_json input.")
        elif payload.agent_id == "churn-analyzer":
            if payload.table_csv:
                churn_result = run_churn_analyzer(payload.table_csv, None, payload.prompt)
                answer = json.dumps(churn_result)
                metadata["source"] = "churn_analyzer_csv"
            else:
                raise HTTPException(status_code=400, detail="churn-analyzer requires CSV input.")
        elif payload.agent_id == "attendance-analyzer":
            if payload.table_csv:
                attendance_result = run_attendance_analyzer(payload.table_csv, None, payload.prompt)
                answer = json.dumps(attendance_result)
                metadata["source"] = "attendance_analyzer_csv"
            else:
                raise HTTPException(status_code=400, detail="attendance-analyzer requires CSV input.")
        elif payload.agent_id == "word-analyst":
            if payload.file_path:
                file_path = payload.file_path
                if not os.path.isabs(file_path):
                    file_path = str(UPLOAD_DIR / file_path)
                if not Path(file_path).exists():
                    raise HTTPException(status_code=400, detail=f"File not found: {payload.file_path}")
                word_result = run_word_analyst(payload.table_csv, payload.table_json, payload.prompt, file_path)
            elif payload.table_csv or payload.table_json:
                word_result = run_word_analyst(payload.table_csv, payload.table_json, payload.prompt)
            else:
                raise HTTPException(status_code=400, detail="word-analyst requires file_path, CSV, or table_json input.")

            if isinstance(word_result, dict):
                answer = (
                    word_result.get("answer")
                    or word_result.get("summary")
                    or word_result.get("message")
                    or json.dumps(word_result, ensure_ascii=False)
                )
                result_metadata = word_result.get("metadata")
                if isinstance(result_metadata, dict):
                    # Serialize datetime objects to ISO format strings
                    result_metadata = serialize_datetime_objects(result_metadata)
                    metadata.update(result_metadata)
                metadata.update({"source": "word_analyst"})

                for download_field in ("edited_docx_path", "generated_docx_path", "download_path"):
                    path_value = word_result.get(download_field)
                    if path_value:
                        try:
                            download_path = Path(path_value).resolve()
                            upload_root = UPLOAD_DIR.resolve()
                            if download_path.is_file() and str(download_path).startswith(str(upload_root)):
                                relative_path = str(download_path.relative_to(upload_root))
                                metadata["download_url"] = f"/api/download-file?file_path={quote_plus(relative_path)}"
                                metadata["download_name"] = download_path.name
                                break
                        except Exception:
                            continue
            else:
                answer = str(word_result)
                metadata.update({"source": "word_analyst"})
        elif payload.agent_id == "ppt-analyst":
            if payload.table_csv or payload.table_json:
                ppt_result = run_ppt_analyst(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(ppt_result)
                metadata["source"] = "ppt_analyst"
            else:
                raise HTTPException(status_code=400, detail="ppt-analyst requires CSV or table_json input.")
        elif payload.agent_id == "email-analyzer":
            if payload.table_csv or payload.table_json:
                email_result = run_email_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(email_result)
                metadata["source"] = "email_analyzer"
            else:
                raise HTTPException(status_code=400, detail="email-analyzer requires CSV or table_json input.")
        elif payload.agent_id == "transcript-analyzer":
            if payload.table_csv or payload.table_json:
                transcript_result = run_transcript_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(transcript_result)
                metadata["source"] = "transcript_analyzer"
            else:
                raise HTTPException(status_code=400, detail="transcript-analyzer requires CSV or table_json input.")
        elif payload.agent_id == "invoice-processor":
            if payload.table_csv or payload.table_json:
                invoice_result = run_invoice_processor(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(invoice_result)
                metadata["source"] = "invoice_processor"
            else:
                raise HTTPException(status_code=400, detail="invoice-processor requires CSV or table_json input.")
        elif payload.agent_id == "recruitment-analyst":
            if payload.table_csv or payload.table_json:
                recruitment_result = run_recruitment_analyst(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(recruitment_result)
                metadata["source"] = "recruitment_analyst"
            else:
                raise HTTPException(status_code=400, detail="recruitment-analyst requires CSV or table_json input.")
        elif payload.agent_id == "performance-review":
            if payload.table_csv or payload.table_json:
                perf_result = run_performance_review_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(perf_result)
                metadata["source"] = "performance_review_analyzer"
            else:
                raise HTTPException(status_code=400, detail="performance-review requires CSV or table_json input.")
        elif payload.agent_id == "budget-actuals":
            if payload.table_csv or payload.table_json:
                budget_result = run_budget_actuals_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(budget_result)
                metadata["source"] = "budget_actuals_analyzer"
            else:
                raise HTTPException(status_code=400, detail="budget-actuals requires CSV or table_json input.")
        elif payload.agent_id == "expense-auditor":
            if payload.table_csv or payload.table_json:
                expense_result = run_expense_auditor(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(expense_result)
                metadata["source"] = "expense_auditor"
            else:
                raise HTTPException(status_code=400, detail="expense-auditor requires CSV or table_json input.")
        elif payload.agent_id == "ar-aging":
            if payload.table_csv or payload.table_json:
                ar_result = run_ar_aging_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(ar_result)
                metadata["source"] = "ar_aging_analyzer"
            else:
                raise HTTPException(status_code=400, detail="ar-aging requires CSV or table_json input.")
        elif payload.agent_id == "cashflow-forecast":
            if payload.table_csv or payload.table_json:
                cashflow_result = run_cashflow_forecast_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(cashflow_result)
                metadata["source"] = "cashflow_forecast_analyzer"
            else:
                raise HTTPException(status_code=400, detail="cashflow-forecast requires CSV or table_json input.")
        elif payload.agent_id == "vendor-spend":
            if payload.table_csv or payload.table_json:
                vendor_result = run_vendor_spend_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(vendor_result)
                metadata["source"] = "vendor_spend_analyzer"
            else:
                raise HTTPException(status_code=400, detail="vendor-spend requires CSV or table_json input.")
        elif payload.agent_id == "payment-optimizer":
            if payload.table_csv or payload.table_json:
                payment_result = run_payment_optimizer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(payment_result)
                metadata["source"] = "payment_optimizer"
            else:
                raise HTTPException(status_code=400, detail="payment-optimizer requires CSV or table_json input.")
        elif payload.agent_id == "project-timeline":
            if payload.table_csv or payload.table_json:
                project_result = run_project_timeline_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(project_result)
                metadata["source"] = "project_timeline_analyzer"
            else:
                raise HTTPException(status_code=400, detail="project-timeline requires CSV or table_json input.")
        elif payload.agent_id == "sla-compliance":
            if payload.table_csv or payload.table_json:
                sla_result = run_sla_compliance_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(sla_result)
                metadata["source"] = "sla_compliance_analyzer"
            else:
                raise HTTPException(status_code=400, detail="sla-compliance requires CSV or table_json input.")
        elif payload.agent_id == "inventory-analyst":
            if payload.table_csv or payload.table_json:
                inventory_result = run_inventory_analyst(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(inventory_result)
                metadata["source"] = "inventory_analyst"
            else:
                raise HTTPException(status_code=400, detail="inventory-analyst requires CSV or table_json input.")
        elif payload.agent_id == "supply-chain":
            if payload.table_csv or payload.table_json:
                supply_result = run_supply_chain_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(supply_result)
                metadata["source"] = "supply_chain_analyzer"
            else:
                raise HTTPException(status_code=400, detail="supply-chain requires CSV or table_json input.")
        elif payload.agent_id == "leads-analyzer":
            if payload.table_csv or payload.table_json:
                leads_result = run_leads_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(leads_result)
                metadata["source"] = "leads_analyzer"
            else:
                raise HTTPException(status_code=400, detail="leads-analyzer requires CSV or table_json input.")
        elif payload.agent_id == "campaign-performance":
            if payload.table_csv or payload.table_json:
                campaign_result = run_campaign_performance_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(campaign_result)
                metadata["source"] = "campaign_performance_analyzer"
            else:
                raise HTTPException(status_code=400, detail="campaign-performance requires CSV or table_json input.")
        elif payload.agent_id == "survey-analyzer":
            if payload.table_csv or payload.table_json:
                survey_result = run_survey_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(survey_result)
                metadata["source"] = "survey_analyzer"
            else:
                raise HTTPException(status_code=400, detail="survey-analyzer requires CSV or table_json input.")
        elif payload.agent_id == "access-rights":
            if payload.table_csv or payload.table_json:
                access_result = run_access_rights_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(access_result)
                metadata["source"] = "access_rights_analyzer"
            else:
                raise HTTPException(status_code=400, detail="access-rights requires CSV or table_json input.")
        elif payload.agent_id == "license-tracker":
            if payload.table_csv or payload.table_json:
                license_result = run_license_tracker_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(license_result)
                metadata["source"] = "license_tracker_analyzer"
            else:
                raise HTTPException(status_code=400, detail="license-tracker requires CSV or table_json input.")
        elif payload.agent_id == "incident-analyzer":
            if payload.table_csv or payload.table_json:
                incident_result = run_incident_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(incident_result)
                metadata["source"] = "incident_analyzer"
            else:
                raise HTTPException(status_code=400, detail="incident-analyzer requires CSV or table_json input.")
        elif payload.agent_id == "ml-modeler":
            if payload.table_csv or payload.table_json:
                ml_result = run_ml_modeler(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(ml_result)
                metadata["source"] = "ml_modeler"
            else:
                raise HTTPException(status_code=400, detail="ml-modeler requires CSV or table_json input.")
        elif payload.agent_id == "log-analyst":
            if payload.table_csv or payload.table_json:
                log_result = run_log_analyst(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(log_result)
                metadata["source"] = "log_analyst"
            else:
                raise HTTPException(status_code=400, detail="log-analyst requires CSV or table_json input.")
        elif payload.agent_id == "image-processor":
            if payload.table_csv or payload.table_json:
                image_result = run_image_processor(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(image_result)
                metadata["source"] = "image_processor"
            else:
                raise HTTPException(status_code=400, detail="image-processor requires CSV or table_json input.")
        elif payload.agent_id == "pdf-extractor":
            if payload.table_csv or payload.table_json:
                pdf_result = run_pdf_extractor(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(pdf_result)
                metadata["source"] = "pdf_extractor"
            else:
                raise HTTPException(status_code=400, detail="pdf-extractor requires CSV or table_json input.")
        elif payload.agent_id == "timeseries-forecaster":
            if payload.table_csv or payload.table_json:
                ts_result = run_timeseries_forecaster(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(ts_result)
                metadata["source"] = "timeseries_forecaster"
            else:
                raise HTTPException(status_code=400, detail="timeseries-forecaster requires CSV or table_json input.")
        elif payload.agent_id == "multifile-correlation":
            if payload.table_csv or payload.table_json:
                multi_result = run_multifile_correlation_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(multi_result)
                metadata["source"] = "multifile_correlation_analyzer"
            else:
                raise HTTPException(status_code=400, detail="multifile-correlation requires CSV or table_json input.")
        elif payload.agent_id == "data-quality":
            if payload.table_csv or payload.table_json:
                quality_result = run_data_quality_analyzer(payload.table_csv, payload.table_json, payload.prompt)
                answer = json.dumps(quality_result)
                metadata["source"] = "data_quality_analyzer"
            else:
                raise HTTPException(status_code=400, detail="data-quality requires CSV or table_json input.")
        elif payload.table_csv or payload.table_json:
            table_data = payload.table_json if payload.table_json is not None else payload.table_csv
            answer = get_langchain_executor().run_dataframe_agent(
                table_data=table_data,
                prompt=payload.prompt,
            )
            metadata["source"] = "dataframe_agent"
        elif payload.mode in ("plan", "preview"):
            preview = get_agent().plan(payload.prompt, top_k=payload.top_k)
            answer = json.dumps({
                "goal": preview.goal,
                "planned_tools": preview.planned_tools,
                "irreversible_tools": preview.irreversible_tools,
                "needs_confirmation": preview.needs_confirmation,
            })
            metadata["source"] = "plan_preview"
        elif payload.mode in ("execute", "run", "workflow"):
            preview = get_agent().plan(payload.prompt, top_k=payload.top_k)
            result = get_agent().execute_plan(preview, confirmed=payload.confirm)
            answer = result.answer
            metadata.update({
                "source": "execute_plan",
                "steps": [step.to_dict() for step in result.steps],
                "partial": result.partial,
                "goal_achieved": result.goal_achieved,
                "duration_ms": getattr(result, "total_duration_ms", None),
            })
        elif payload.mode == "report":
            report_result = get_agent().generate_report(payload.prompt, top_k=payload.top_k)
            answer = json.dumps(report_result.to_dict())
            metadata["source"] = "report"
        elif payload.mode in ("graph", "langgraph"):
            graph_result = get_langchain_executor().run_langgraph_workflow(payload.agent_id, payload.prompt)
            answer = graph_result.get("result", "")
            metadata.update({"source": "langgraph_workflow", "graph_result": graph_result})
        elif payload.use_langchain:
            answer = get_langchain_executor().chat(payload.prompt, payload.agent_id, max_tokens=1024)
            metadata["source"] = "langchain_chat"
        else:
            query_result = get_agent().query(payload.prompt, top_k=payload.top_k)
            answer = query_result.answer
            metadata.update({
                "source": "query",
                "query": query_result.query,
                "chunks": [chunk.to_dict() for chunk in query_result.chunks],
            })
        execution_record.status = ExecutionStatus.COMPLETED
        execution_record.output.answer = answer
        execution_record.output.summary = answer
        execution_record.metadata.update(metadata)
        execution_record.updated_at = datetime.now(timezone.utc).isoformat()
        execution_record.completed_at = execution_record.updated_at
    except HTTPException as exc:
        execution_record.status = ExecutionStatus.FAILED
        execution_record.error = str(exc.detail)
        execution_record.updated_at = datetime.now(timezone.utc).isoformat()
        execution_record.completed_at = execution_record.updated_at
        raise
    except ConfirmationRequiredError as exc:
        logger.info(
            "Agent execution requires confirmation for %s: %s",
            payload.agent_id,
            exc.irreversible_tools,
        )
        execution_record.status = ExecutionStatus.WAITING_FOR_APPROVAL
        execution_record.error = str(exc)
        execution_record.updated_at = datetime.now(timezone.utc).isoformat()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "confirmation_required",
                "message": str(exc),
                "irreversible_tools": exc.irreversible_tools,
                "agent_id": payload.agent_id,
            },
        ) from exc
    except Exception as exc:
        execution_record.status = ExecutionStatus.FAILED
        execution_record.error = str(exc)
        execution_record.updated_at = datetime.now(timezone.utc).isoformat()
        execution_record.completed_at = execution_record.updated_at
        logger.error(f"Agent run failed for {payload.agent_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if downloaded_file_path is not None:
            try:
                downloaded_file_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove temporary uploaded file: %s", exc)

    # Ensure answer is a string for the response model
    if not isinstance(answer, str):
        try:
            answer = json.dumps(answer)
        except Exception:
            answer = str(answer)

    return AgentExecutionResponse(
        agent_id=payload.agent_id,
        mode=payload.mode,
        answer=answer,
        execution_id=execution_record.execution_id,
        trace_id=execution_record.trace_id,
        status=execution_record.status.value,
        metadata=metadata,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    category: Optional[str] = Form("General"),
    user_goal: Optional[str] = Form(None),
    priority: Optional[str] = Form("Medium"),
    due_date: Optional[str] = Form(None),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    # Basic validation limits
    max_upload_bytes = _max_upload_bytes()
    ALLOWED_EXT = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".db", ".sqlite"}

    ingested: List[Dict[str, Any]] = []
    metadata = {
        "category": category,
        "user_goal": user_goal or "",
        "priority": priority,
        "due_date": due_date,
    }

    for upload in files:
        file_name = Path(upload.filename).name
        file_path = UPLOAD_DIR / file_name
        ext = Path(file_name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}")
        try:
            await _write_upload_to_path(upload, file_path, max_upload_bytes)
            asset = get_agent().ingest_file(str(file_path), metadata)
            ingested.append(asset.to_dict())
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Failed to ingest {file_name}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    return {"ingested": ingested}


async def _write_upload_to_path(upload: UploadFile, file_path: Path, max_bytes: int) -> int:
    total_bytes = 0
    try:
        with file_path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the configured upload limit of {max_bytes} bytes.",
                    )
                destination.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return total_bytes
    except Exception:
        file_path.unlink(missing_ok=True)
        raise


async def _save_uploaded_file(file: UploadFile) -> Dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    original_name = Path(file.filename).name
    suffix = Path(original_name).suffix.lower()
    file_name = f"{uuid.uuid4().hex}{suffix}"
    file_path = UPLOAD_DIR / file_name

    # Extension validation for data files
    ext = suffix
    ALLOWED_EXT = {".db", ".sqlite", ".sqlite3", ".csv", ".xlsx", ".xls", ".json", ".docx", ".doc", ".pdf"}
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}. Allowed: {ALLOWED_EXT}")

    await _write_upload_to_path(file, file_path, _max_upload_bytes())
    return {
        "file_path": file_name,
        "absolute_path": str(file_path),
        "original_name": original_name,
        "message": f"File uploaded successfully"
    }


@app.post("/upload-file")
async def upload_file_simple(file: UploadFile = File(...)) -> Dict[str, str]:
    return await _save_uploaded_file(file)


@app.post("/upload")
async def upload_file_alias(file: UploadFile = File(...)) -> Dict[str, str]:
    return await _save_uploaded_file(file)


@app.get("/download-file")
async def download_file(file_path: str) -> FileResponse:
    resolved_root = UPLOAD_DIR.resolve()
    requested_path = (UPLOAD_DIR / file_path).resolve()
    if resolved_root not in requested_path.parents and requested_path != resolved_root:
        raise HTTPException(status_code=403, detail="Forbidden file path")
    if not requested_path.exists() or not requested_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(requested_path),
        filename=requested_path.name,
        media_type="application/octet-stream",
    )


@app.post("/plan", response_model=PlanPreviewResponse)
async def plan_agent(payload: PlanRequest) -> PlanPreviewResponse:
    if not payload.goal or not payload.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")

    preview = get_agent().plan(payload.goal, top_k=payload.top_k)
    return {
        "goal": preview.goal,
        "planned_tools": preview.planned_tools,
        "irreversible_tools": preview.irreversible_tools,
        "needs_confirmation": preview.needs_confirmation,
        "chunk_count": len(preview.chunks),
    }


@app.post("/execute-plan", response_model=PlanExecutionResponse)
async def execute_plan(payload: ExecutePlanRequest) -> PlanExecutionResponse:
    if not payload.goal or not payload.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")

    plan_preview = get_agent().plan(payload.goal, top_k=payload.top_k)
    try:
        result = get_agent().execute_plan(plan_preview, confirmed=payload.confirm)
        return result.to_dict()
    except ConfirmationRequiredError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/report", response_model=ReportResponse)
async def build_report(payload: ReportRequest) -> ReportResponse:
    if not payload.goal or not payload.goal.strip():
        raise HTTPException(status_code=400, detail="Report goal cannot be empty.")

    result = get_agent().generate_report(payload.goal, query=payload.query, top_k=payload.top_k)
    return result.to_dict()


@app.post("/report/execute-block")
async def execute_report_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    block_id = payload.get("block_id")
    code = payload.get("code")
    section_name = payload.get("section_name") or payload.get("section") or ""

    if not block_id:
        raise HTTPException(status_code=400, detail="Missing block_id")

    # Build a minimal CodeBlock object and execute via ReportBuilder
    try:
        block = CodeBlock(block_id=str(block_id), section_name=section_name, code=code or "")
        updated = get_agent().reporter.execute_code_block(block, edited_code=code)
        return {"code_block": updated.to_dict()}
    except Exception as exc:
        logger.error(f"Failed to execute code block {block_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/report/finalize", response_model=ReportResponse)
async def finalize_report(payload: Dict[str, Any]) -> ReportResponse:
    # Expect full report payload from frontend (sections, code blocks, goal)
    try:
        goal = payload.get("goal", "Report")
        sections_payload = payload.get("sections", [])
        sections: List[ReportSection] = []
        pending_blocks: List[CodeBlock] = []

        for s in sections_payload:
            cb = None
            if s.get("code_block"):
                c = s.get("code_block")
                cb = CodeBlock(
                    block_id=c.get("block_id", ""),
                    section_name=s.get("title", ""),
                    code=c.get("code", ""),
                    status=c.get("status", "pending"),
                    stdout=c.get("stdout"),
                    stderr=c.get("stderr"),
                    images=c.get("images") or [],
                    include_in_report=c.get("include_in_report", True),
                )
                if cb.status in ("pending",):
                    pending_blocks.append(cb)

            section = ReportSection(
                title=s.get("title", ""),
                content=s.get("content", ""),
                code_block=cb,
                chart_base64=s.get("chart_base64"),
            )
            sections.append(section)

        report_obj = BuiltReportResult(
            goal=goal,
            sections=sections,
            pending_code_blocks=pending_blocks,
            html=payload.get("html"),
            ready_to_finalize=len(pending_blocks) == 0,
        )

        finalized = get_agent().reporter.finalize(report_obj)
        return finalized.to_dict()
    except Exception as exc:
        logger.error(f"Failed to finalize report: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/execute-plan/stream")
async def execute_plan_stream(goal: str, top_k: int = 5):
    if not goal or not goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")

    # Obtain planning context
    preview = get_agent().plan(goal, top_k=top_k)

    # Use planner internals to create PlanStep list
    plan_steps = get_agent().planner._plan_phase(goal, preview.chunks)

    def event(data: Any) -> bytes:
        payload = json.dumps(data, default=str)
        return f"data: {payload}\n\n".encode("utf-8")

    def gen():
        # Send initial preview
        yield event({"type": "preview", "planned_tools": preview.planned_tools, "irreversible": preview.irreversible_tools})

        completed: Dict[int, Any] = {}

        # Execute steps sequentially and stream updates
        for step in plan_steps:
            try:
                # notify running
                yield event({"type": "step", "step": step.step_num, "tool": step.tool_name, "status": "running", "rationale": step.rationale})

                if step.tool_name is None:
                    # direct answer
                    answer = get_agent().llm.generate(f"Answer for goal: {goal}")
                    step.result = {"answer": answer}
                    step.status = "success"
                else:
                    # execute tool via MCP
                    try:
                        result = get_agent().mcp.call_tool(step.tool_name, **(step.args or {}))
                        step.result = result
                        step.status = "success"
                    except Exception as exc:
                        step.status = "error"
                        step.error_msg = str(exc)

                completed[step.step_num] = step

                # stream step result
                yield event({"type": "step_result", "step": step.step_num, "tool": step.tool_name, "status": step.status, "result": step.result, "error": step.error_msg})
            except Exception as exc:
                yield event({"type": "step_result", "step": step.step_num, "tool": getattr(step, 'tool_name', None), "status": "error", "error": str(exc)})

        # Verification / finalization
        verify = get_agent().planner._verify_phase(goal, preview.chunks, {s.step_num: s for s in completed.values()})
        yield event({"type": "done", "answer": verify.answer, "goal_achieved": verify.goal_achieved, "steps": [s.to_dict() for s in completed.values()]})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/documents", response_model=List[DocumentAssetResponse])
async def list_documents(category: Optional[str] = None) -> List[DocumentAssetResponse]:
    docs = get_agent().documents.list_documents(category)
    return [doc.to_dict() for doc in docs]


@app.post("/tools/credentials")
async def update_tool_credentials(
    payload: ToolCredentialsRequest,
) -> Dict[str, Any]:
    if not payload.credentials:
        raise HTTPException(status_code=400, detail="No credentials provided.")
    try:
        ToolCredentialStore.update(payload.credentials)
    except Exception as exc:
        logger.error(f"Failed to save tool credentials: {exc}")
        raise HTTPException(status_code=500, detail="Unable to save tool credentials.")
    return {"status": "ok", "updated_keys": sorted(payload.credentials.keys())}


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(**get_agent().get_status())
