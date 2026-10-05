#!/usr/bin/env python3
import traceback

# Modules corresponding to failed agent imports
modules = [
    "access_rights_analyzer",
    "ar_aging_analyzer",
    "budget_actuals_analyzer",
    "campaign_performance_analyzer",
    "cashflow_forecast_analyzer",
    "data_quality_analyzer",
    "financial_data_analyst",
    "license_tracker_analyzer",
    "multifile_correlation_analyzer",
    "payroll_analyst",
    "performance_review_analyzer",
    "project_timeline_analyzer",
    "sales_pipeline_analyst",
    "sla_compliance_analyzer",
    "supply_chain_analyzer",
    "vendor_spend_analyzer",
]

print("Import debug for failing modules:\n")
for mod in modules:
    try:
        __import__(mod)
        print(f"✓ {mod} imported successfully")
    except Exception as e:
        print(f"✗ {mod} FAILED to import")
        print(traceback.format_exc())
        print("-"*60)

print("Done.")
