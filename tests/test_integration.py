#!/usr/bin/env python3
"""
Integration Test: Frontend Agent IDs vs Backend Agent Registry
This script verifies that frontend agent definitions are aligned with the backend registry
and that the backend API contains execution branches for every frontend agent.
"""

import sys
import re
from pathlib import Path

FRONTEND_AGENTS_PATH = Path(__file__).resolve().parent.parent / "Microfinince  frontend" / "lib" / "agents.ts"
BACKEND_API_PATH = Path(__file__).resolve().parent / "backend_api.py"

EXPECTED_BACKEND_MODULES = {
    "word-analyst": "word_analyst_agent",
    "ppt-analyst": "ppt_analyst_agent",
    "email-analyzer": "email_analyzer_agent",
    "transcript-analyzer": "transcript_analyzer_agent",
    "invoice-processor": "invoice_processor_agent",
    "payroll-analyst": "payroll_analyst",
    "attendance-analyzer": "attendance_analyzer_agent",
    "recruitment-analyst": "recruitment_analyst_agent",
    "performance-review": "performance_review_analyzer",
    "budget-actuals": "budget_actuals_analyzer",
    "expense-auditor": "expense_auditor_agent",
    "ar-aging": "ar_aging_analyzer",
    "cashflow-forecast": "cashflow_forecast_analyzer",
    "payment-optimizer": "payment_optimizer_agent",
    "vendor-spend": "vendor_spend_analyzer",
    "project-timeline": "project_timeline_analyzer",
    "sla-compliance": "sla_compliance_analyzer",
    "inventory-analyst": "inventory_analyst_agent",
    "supply-chain": "supply_chain_analyzer",
    "sales-pipeline": "sales_pipeline_analyst",
    "churn-analyzer": "churn_analyzer_agent",
    "campaign-performance": "campaign_performance_analyzer",
    "survey-analyzer": "survey_analyzer_agent",
    "access-rights": "access_rights_analyzer",
    "license-tracker": "license_tracker_analyzer",
    "incident-analyzer": "incident_analyzer_agent",
    "csv-analyst": "csv_analyst_agent",
    "sql-analyst": "sql_analyst_agent",
    "excel-analyst": "excel_analyst_agent",
    "financial-data": "financial_data_analyst",
    "ml-modeler": "ml_modeler_agent",
    "log-analyst": "log_analyst_agent",
    "image-processor": "image_processor_agent",
    "pdf-extractor": "pdf_extractor_agent",
    "json-analyst": "json_analyst_agent",
    "timeseries-forecaster": "timeseries_forecaster_agent",
    "multifile-correlation": "multifile_correlation_analyzer",
    "data-quality": "data_quality_analyzer",
}

CATEGORY_LABELS = {
    "docs": "Documents",
    "hr": "HR & People",
    "finance": "Finance",
    "ops": "Operations",
    "sales": "Sales & Marketing",
    "it": "IT & Compliance",
    "data": "Data & Analytics",
}


def load_frontend_agents() -> list[dict[str, str]]:
    if not FRONTEND_AGENTS_PATH.exists():
        raise FileNotFoundError(
            f"Frontend agent registry not found: {FRONTEND_AGENTS_PATH}"
        )

    text = FRONTEND_AGENTS_PATH.read_text(encoding="utf-8")
    start = text.find("export const AGENTS: Agent[] = [")
    if start == -1:
        raise ValueError("Could not locate AGENTS definition in frontend lib/agents.ts")

    end = text.find("];", start)
    if end == -1:
        raise ValueError("Could not locate end of AGENTS definition in frontend lib/agents.ts")

    block = text[start:end]
    matches = re.findall(
        r'id:\s*"([a-z0-9\-]+)"[\s\S]*?categoryId:\s*"([a-z0-9\-]+)"',
        block,
    )
    return [{"id": agent_id, "categoryId": category_id} for agent_id, category_id in matches]


def load_backend_api_agent_ids() -> list[str]:
    text = BACKEND_API_PATH.read_text(encoding="utf-8")
    ids = re.findall(r'elif payload\.agent_id == "([a-z0-9\-]+)"', text)
    first_if = re.search(r'if payload\.agent_id == "([a-z0-9\-]+)"', text)
    if first_if:
        ids.insert(0, first_if.group(1))
    return ids


def format_category_summary(agent_ids: list[str]) -> str:
    sorted_ids = sorted(agent_ids)
    return ", ".join(sorted_ids)


def main() -> int:
    print("=" * 80)
    print("INTEGRATION TEST: Frontend-Backend Agent Alignment")
    print("=" * 80)
    print()

    frontend_agents = load_frontend_agents()
    frontend_ids = [agent["id"] for agent in frontend_agents]
    backend_runner_ids = load_backend_api_agent_ids()
    backend_registry_ids = list(EXPECTED_BACKEND_MODULES.keys())

    frontend_set = set(frontend_ids)
    backend_run_set = set(backend_runner_ids)
    backend_registry_set = set(backend_registry_ids)

    print(f"[OK] Frontend Agent IDs Loaded: {len(frontend_ids)}")
    print(f"[OK] Backend Agent Execution Branches: {len(backend_runner_ids)}")
    print(f"[OK] Backend Agent Registry Candidates: {len(backend_registry_ids)}")
    print()

    print("Agent Alignment Checks:")
    print("-" * 80)

    unregistered = sorted(frontend_set - backend_registry_set)
    if unregistered:
        print(f"[FAIL] Frontend agent IDs missing from backend registry: {len(unregistered)}")
        for agent_id in unregistered:
            print(f"  - {agent_id}")
        print()
    else:
        print("[OK] All frontend agent IDs are present in the backend registry.")
        print()

    unhandled = sorted(frontend_set - backend_run_set)
    if unhandled:
        print(f"[FAIL] Frontend agent IDs missing backend execution branches: {len(unhandled)}")
        for agent_id in unhandled:
            print(f"  - {agent_id}")
        print()
    else:
        print("[OK] All frontend agent IDs have a backend /agent/run branch.")
        print()

    extra_registry = sorted(backend_registry_set - frontend_set)
    if extra_registry:
        print(f"[INFO] Backend registry contains extra agent IDs not exposed in frontend: {len(extra_registry)}")
        for agent_id in extra_registry:
            print(f"  - {agent_id}")
        print()
    else:
        print("[OK] No extra backend registry IDs outside frontend agent list.")
        print()

    extra_runner = sorted(backend_run_set - frontend_set)
    if extra_runner:
        print(f"[INFO] Backend execution branches contain extra agent IDs not in frontend: {len(extra_runner)}")
        for agent_id in extra_runner:
            print(f"  - {agent_id}")
        print()
    else:
        print("[OK] No extra backend execution branches outside frontend agent list.")
        print()

    print("Frontend Agent ID Summary:")
    print("-" * 80)
    category_map: dict[str, list[str]] = {key: [] for key in CATEGORY_LABELS}
    for agent in frontend_agents:
        category = agent["categoryId"]
        if category in category_map:
            category_map[category].append(agent["id"])
        else:
            category_map.setdefault(category, []).append(agent["id"])

    for category, label in CATEGORY_LABELS.items():
        category_ids = sorted(category_map.get(category, []))
        if category_ids:
            print(f"{label:20}: {format_category_summary(category_ids)}")
    print()

    print("=" * 80)
    if unregistered or unhandled:
        print("[FAIL] INTEGRATION ISSUES FOUND. Please synchronize frontend and backend agents.")
        return 1

    print("[OK] INTEGRATION READY: Frontend and backend agent wiring are aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
