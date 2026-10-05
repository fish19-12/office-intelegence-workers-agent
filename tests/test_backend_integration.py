#!/usr/bin/env python3
"""
Backend Integration Test: Verify all agents are importable and correctly wired.
"""

import sys
import os
import json

# Suppress embeddings download output during import
os.environ["HF_HUB_QUIET"] = "1"

def test_imports():
    """Test that all agent modules can be imported."""
    print("Testing Agent Module Imports...")
    print("-" * 80)
    
    agents_to_test = [
        "agents.analytics.csv_analyst_agent",
        "agents.analytics.sql_analyst_agent",
        "agents.analytics.excel_analyst_agent",
        "agents.finance.financial_data_analyst",
        "agents.finance.payroll_analyst",
        "agents.sales_marketing.sales_pipeline_analyst",
        "agents.analytics.json_analyst_agent",
        "agents.people.churn_analyzer_agent",
        "agents.people.attendance_analyzer_agent",
        "agents.documents.word_analyst_agent",
        "agents.documents.ppt_analyst_agent",
        "agents.documents.email_analyzer_agent",
        "agents.documents.transcript_analyzer_agent",
        "agents.finance.invoice_processor_agent",
        "agents.people.recruitment_analyst_agent",
        "agents.people.performance_review_analyzer",
        "agents.finance.budget_actuals_analyzer",
        "agents.finance.expense_auditor_agent",
        "agents.finance.ar_aging_analyzer",
        "agents.finance.cashflow_forecast_analyzer",
        "agents.finance.vendor_spend_analyzer",
        "agents.finance.payment_optimizer_agent",
        "agents.operations.project_timeline_analyzer",
        "agents.operations.sla_compliance_analyzer",
        "agents.operations.inventory_analyst_agent",
        "agents.operations.supply_chain_analyzer",
        "agents.sales_marketing.leads_analyzer_agent",
        "agents.sales_marketing.campaign_performance_analyzer",
        "agents.analytics.survey_analyzer_agent",
        "agents.operations.access_rights_analyzer",
        "agents.operations.license_tracker_analyzer",
        "agents.operations.incident_analyzer_agent",
        "agents.analytics.ml_modeler_agent",
        "agents.operations.log_analyst_agent",
        "agents.documents.image_processor_agent",
        "agents.documents.pdf_extractor_agent",
        "agents.analytics.timeseries_forecaster_agent",
        "agents.analytics.multifile_correlation_analyzer",
        "agents.analytics.data_quality_analyzer",
    ]
    
    failed = []
    for agent_module in agents_to_test:
        try:
            __import__(agent_module)
            print(f"✓ {agent_module}")
        except Exception as e:
            print(f"✗ {agent_module}: {str(e)[:60]}")
            failed.append(agent_module)
    
    print()
    print(f"Results: {len(agents_to_test) - len(failed)}/{len(agents_to_test)} imports successful")
    
    if failed:
        print(f"Failed imports: {', '.join(failed)}")
        return False
    
    return True


def test_registry():
    """Test that agents.core.backend_agent_registry contains all expected agents."""
    print()
    print("Testing Agent Registry...")
    print("-" * 80)
    
    from agents.core.backend_agent_registry import AGENT_REGISTRY, is_known_agent
    
    expected_agents = [
        "csv-analyst",
        "sql-analyst",
        "excel-analyst",
        "financial-data",
        "payroll-analyst",
        "sales-pipeline",
        "json-analyst",
        "churn-analyzer",
        "attendance-analyzer",
        "word-analyst",
        "ppt-analyst",
        "email-analyzer",
        "transcript-analyzer",
        "invoice-processor",
        "recruitment-analyst",
        "performance-review",
        "budget-actuals",
        "expense-auditor",
        "ar-aging",
        "cashflow-forecast",
        "vendor-spend",
        "payment-optimizer",
        "project-timeline",
        "sla-compliance",
        "inventory-analyst",
        "supply-chain",
        "leads-analyzer",
        "campaign-performance",
        "survey-analyzer",
        "access-rights",
        "license-tracker",
        "incident-analyzer",
        "ml-modeler",
        "log-analyst",
        "image-processor",
        "pdf-extractor",
        "timeseries-forecaster",
        "multifile-correlation",
        "data-quality",
    ]
    
    missing = []
    for agent_id in expected_agents:
        if not is_known_agent(agent_id):
            print(f"✗ {agent_id} not in registry")
            missing.append(agent_id)
        else:
            info = AGENT_REGISTRY[agent_id]
            print(f"✓ {agent_id:25} - {info['name']}")
    
    print()
    print(f"Results: {len(expected_agents) - len(missing)}/{len(expected_agents)} agents in registry")
    
    if missing:
        print(f"Missing agents: {', '.join(missing)}")
        return False
    
    return True


def test_backend_api_imports():
    """Test that backend_api can be imported without errors."""
    print()
    print("Testing Backend API Imports...")
    print("-" * 80)
    
    try:
        import backend_api
        print("✓ backend_api module imported successfully")
        
        # Check if FastAPI app exists
        if hasattr(backend_api, "app"):
            print("✓ FastAPI app object found")
            
            # Check routes
            routes = [route.path for route in backend_api.app.routes]
            expected_routes = ["/agent/run", "/upload", "/preview", "/health"]
            for route in expected_routes:
                if any(route in r for r in routes):
                    print(f"✓ Route {route} exists")
                else:
                    print(f"✗ Route {route} not found")
            
            return True
        else:
            print("✗ FastAPI app object not found")
            return False
            
    except Exception as e:
        print(f"✗ Failed to import backend_api: {str(e)[:100]}")
        return False


def main():
    print()
    print("=" * 80)
    print("BACKEND INTEGRATION VERIFICATION")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Agent Module Imports", test_imports()))
    
    # Test 2: Registry
    results.append(("Agent Registry", test_registry()))
    
    # Test 3: Backend API
    results.append(("Backend API", test_backend_api_imports()))
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print()
        print("✓ ALL TESTS PASSED - System is ready for frontend integration!")
        return 0
    else:
        print()
        print("✗ SOME TESTS FAILED - See details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
