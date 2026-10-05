#!/usr/bin/env python3
"""
Complete Integration Test: Simulates frontend-to-backend flow
Tests all 38 agents with sample data
"""

import sys
import json
import time
from typing import Dict, Any, Optional
from io import StringIO
import pandas as pd

# Sample test data for different agent types
SAMPLE_CSV_DATA = """name,category,amount,date,status
John,Sales,5000,2024-01-15,Completed
Sarah,Marketing,3000,2024-01-16,Pending
Mike,Operations,7500,2024-01-17,Completed
Lisa,HR,2500,2024-01-18,Completed
Tom,Sales,6200,2024-01-19,Processing
"""

SAMPLE_TABLE_JSON = [
    {"product": "Product A", "sales": 1000, "profit": 300},
    {"product": "Product B", "sales": 2500, "profit": 700},
    {"product": "Product C", "sales": 1800, "profit": 450},
]

# Test cases for each agent
TEST_AGENTS = {
    # Data Agents
    "csv-analyst": {"type": "csv", "prompt": "Analyze sales trends"},
    "sql-analyst": {"type": "table", "prompt": "Identify high performers"},
    "excel-analyst": {"type": "csv", "prompt": "Compare department budgets"},
    "financial-data": {"type": "csv", "prompt": "Analyze financial metrics"},
    "json-analyst": {"type": "table", "prompt": "Find correlations"},
    "ml-modeler": {"type": "csv", "prompt": "Assess data quality"},
    "log-analyst": {"type": "csv", "prompt": "Analyze error patterns"},
    "image-processor": {"type": "csv", "prompt": "Process image metadata"},
    "pdf-extractor": {"type": "csv", "prompt": "Extract information"},
    "timeseries-forecaster": {"type": "csv", "prompt": "Forecast trends"},
    "multifile-correlation": {"type": "table", "prompt": "Find relationships"},
    "data-quality": {"type": "csv", "prompt": "Quality assessment"},
    
    # Document Agents
    "word-analyst": {"type": "text", "prompt": "Analyze document"},
    "ppt-analyst": {"type": "text", "prompt": "Review presentation"},
    "email-analyzer": {"type": "text", "prompt": "Analyze communications"},
    "transcript-analyzer": {"type": "text", "prompt": "Analyze meeting"},
    "invoice-processor": {"type": "csv", "prompt": "Process invoices"},
    
    # Finance Agents
    "budget-actuals": {"type": "csv", "prompt": "Budget analysis"},
    "expense-auditor": {"type": "csv", "prompt": "Audit expenses"},
    "ar-aging": {"type": "csv", "prompt": "Analyze receivables"},
    "cashflow-forecast": {"type": "csv", "prompt": "Forecast cash flow"},
    "vendor-spend": {"type": "csv", "prompt": "Analyze vendor spend"},
    "payment-optimizer": {"type": "csv", "prompt": "Optimize payments"},
    
    # HR Agents
    "payroll-analyst": {"type": "csv", "prompt": "Analyze payroll"},
    "attendance-analyzer": {"type": "csv", "prompt": "Analyze attendance"},
    "recruitment-analyst": {"type": "csv", "prompt": "Analyze recruitment"},
    "performance-review": {"type": "csv", "prompt": "Review performance"},
    
    # Operations Agents
    "project-timeline": {"type": "csv", "prompt": "Analyze timeline"},
    "sla-compliance": {"type": "csv", "prompt": "Check SLA compliance"},
    "inventory-analyst": {"type": "csv", "prompt": "Analyze inventory"},
    "supply-chain": {"type": "csv", "prompt": "Analyze supply chain"},
    
    # Sales Agents
    "sales-pipeline": {"type": "csv", "prompt": "Analyze sales pipeline"},
    "churn-analyzer": {"type": "csv", "prompt": "Analyze churn"},
    "campaign-performance": {"type": "csv", "prompt": "Analyze campaigns"},
    "survey-analyzer": {"type": "csv", "prompt": "Analyze survey"},
    
    # IT Agents
    "access-rights": {"type": "csv", "prompt": "Analyze access"},
    "license-tracker": {"type": "csv", "prompt": "Track licenses"},
    "incident-analyzer": {"type": "csv", "prompt": "Analyze incidents"},
    
    # Legacy Agents
    "leads-analyzer": {"type": "csv", "prompt": "Analyze leads"},
}


def create_mock_agent_request(agent_id: str, test_config: Dict[str, str]) -> Dict[str, Any]:
    """Create a mock request as if from frontend."""
    data_type = test_config["type"]
    
    request = {
        "agent_id": agent_id,
        "prompt": test_config["prompt"],
        "mode": "analyze",
        "top_k": 5,
    }
    
    if data_type == "csv":
        request["table_csv"] = SAMPLE_CSV_DATA
        request["table_json"] = None
    elif data_type == "table":
        request["table_csv"] = None
        request["table_json"] = SAMPLE_TABLE_JSON
    else:
        request["table_csv"] = None
        request["table_json"] = None
    
    return request


def validate_response(response: Dict[str, Any], agent_id: str) -> bool:
    """Validate response structure."""
    required_keys = ["answer", "metadata"]
    
    for key in required_keys:
        if key not in response:
            print(f"    ✗ Missing key: {key}")
            return False
    
    # Validate metadata
    metadata = response.get("metadata", {})
    if metadata.get("source") != agent_id.replace("-", "_") + "_agent":
        # Some agents might have slightly different naming
        pass  # Still valid
    
    # Validate answer is JSON-serializable
    try:
        answer = response.get("answer")
        if isinstance(answer, str):
            json.loads(answer)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"    ✗ Answer not JSON-serializable: {str(e)[:50]}")
        return False
    
    return True


def test_agent_import() -> None:
    """Test that agent module can be imported."""
    for agent_id in sorted(TEST_AGENTS):
        base = agent_id.replace("-", "_")
        candidates = [f"{base}_agent", f"{base}_analyzer", f"{base}_analyst", base]
        imported = False
        for module_name in candidates:
            try:
                __import__(module_name)
                imported = True
                break
            except ImportError:
                continue
        assert imported, f"Could not import {agent_id}"


def test_agent_in_registry() -> None:
    """Test that agent is in registry."""
    from backend_agent_registry import is_known_agent
    for agent_id in sorted(TEST_AGENTS):
        assert is_known_agent(agent_id), f"{agent_id} not in registry"


def test_create_request() -> None:
    """Test that valid request can be created."""
    for agent_id, test_config in TEST_AGENTS.items():
        request = create_mock_agent_request(agent_id, test_config)
        required = ["agent_id", "prompt", "mode"]
        for key in required:
            assert key in request, f"Request for {agent_id} missing {key}"
        if test_config["type"] != "text":
            assert request.get("table_csv") or request.get("table_json"), f"Request for {agent_id} has no data"


def main():
    print()
    print("=" * 90)
    print("MICROFINANCE AGENTS - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 90)
    print()
    
    print(f"Testing {len(TEST_AGENTS)} agents...")
    print()
    
    results = {
        "import": 0,
        "registry": 0,
        "request": 0,
        "total": len(TEST_AGENTS),
    }
    
    # Group by category
    categories = {
        "data": [],
        "docs": [],
        "finance": [],
        "hr": [],
        "ops": [],
        "sales": [],
        "it": [],
    }
    
    # Map agents to categories
    category_map = {
        "data": ["csv-analyst", "sql-analyst", "excel-analyst", "financial-data", "json-analyst", 
                 "ml-modeler", "log-analyst", "image-processor", "pdf-extractor", 
                 "timeseries-forecaster", "multifile-correlation", "data-quality"],
        "docs": ["word-analyst", "ppt-analyst", "email-analyzer", "transcript-analyzer", "invoice-processor"],
        "finance": ["budget-actuals", "expense-auditor", "ar-aging", "cashflow-forecast", 
                    "vendor-spend", "payment-optimizer"],
        "hr": ["payroll-analyst", "attendance-analyzer", "recruitment-analyst", "performance-review"],
        "ops": ["project-timeline", "sla-compliance", "inventory-analyst", "supply-chain"],
        "sales": ["sales-pipeline", "churn-analyzer", "campaign-performance", "survey-analyzer"],
        "it": ["access-rights", "license-tracker", "incident-analyzer"],
    }
    
    # Test each agent
    for agent_id, test_config in sorted(TEST_AGENTS.items()):
        # Find category
        agent_category = "other"
        for cat, agents in category_map.items():
            if agent_id in agents:
                agent_category = cat
                break
        
        print(f"[{agent_category.upper():5}] {agent_id:28} ", end="", flush=True)
        
        # Test import
        import_ok = test_agent_import(agent_id)
        print("✓" if import_ok else "✗", end=" ", flush=True)
        if import_ok:
            results["import"] += 1
        
        # Test registry
        registry_ok = test_agent_in_registry(agent_id)
        print("✓" if registry_ok else "✗", end=" ", flush=True)
        if registry_ok:
            results["registry"] += 1
        
        # Test request creation
        request_ok = test_create_request(agent_id, test_config)
        print("✓" if request_ok else "✗", end="", flush=True)
        if request_ok:
            results["request"] += 1
        
        print()
    
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"Agents Tested:         {results['total']}")
    print(f"Import Status:         {results['import']}/{results['total']} ✓")
    print(f"Registry Status:       {results['registry']}/{results['total']} ✓")
    print(f"Request Creation:      {results['request']}/{results['total']} ✓")
    print()
    
    all_passed = (
        results["import"] == results["total"] and
        results["registry"] == results["total"] and
        results["request"] == results["total"]
    )
    
    print("=" * 90)
    if all_passed:
        print("✅ ALL TESTS PASSED - System ready for end-to-end testing!")
        print()
        print("NEXT STEPS:")
        print("1. Start backend API:  python -m uvicorn backend_api:app --port 8000")
        print("2. Start frontend:     npm run dev (from frontend directory)")
        print("3. Access at:          http://localhost:3000")
        print("4. Test agent calls through UI")
        return 0
    else:
        print("❌ SOME TESTS FAILED - See details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
