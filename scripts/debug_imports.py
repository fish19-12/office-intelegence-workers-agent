#!/usr/bin/env python3
import traceback

# Modules corresponding to failed agent imports
modules = [
    "agents.operations.access_rights_analyzer",
    "agents.finance.ar_aging_analyzer",
    "agents.finance.budget_actuals_analyzer",
    "agents.sales_marketing.campaign_performance_analyzer",
    "agents.finance.cashflow_forecast_analyzer",
    "agents.analytics.data_quality_analyzer",
    "agents.finance.financial_data_analyst",
    "agents.operations.license_tracker_analyzer",
    "agents.analytics.multifile_correlation_analyzer",
    "agents.finance.payroll_analyst",
    "agents.people.performance_review_analyzer",
    "agents.operations.project_timeline_analyzer",
    "agents.sales_marketing.sales_pipeline_analyst",
    "agents.operations.sla_compliance_analyzer",
    "agents.operations.supply_chain_analyzer",
    "agents.finance.vendor_spend_analyzer",
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
