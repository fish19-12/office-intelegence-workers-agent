from __future__ import annotations

from typing import Dict, Optional

# Backend registry for frontend agent IDs.
# This makes it easier to validate requests and provide agent-specific prompts.
AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
    # Documents
    "word-analyst": {
        "name": "Word Document Analyst",
        "description": "Analyze .docx files and extract document insights.",
    },
    "ppt-analyst": {
        "name": "PowerPoint Analyst",
        "description": "Analyze .pptx slide decks and summarize slide content.",
    },
    "email-analyzer": {
        "name": "Email Batch Analyzer",
        "description": "Analyze email exports and identify trends or issues.",
    },
    "transcript-analyzer": {
        "name": "Meeting Transcript Analyzer",
        "description": "Process meeting transcripts and extract action items.",
    },
    "invoice-processor": {
        "name": "Invoice & Receipt Processor",
        "description": "Extract and analyze invoice and receipt data.",
    },

    # HR & People
    "recruitment-analyst": {
        "name": "Recruitment Pipeline Analyst",
        "description": "Analyze recruitment funnel metrics and hiring trends.",
    },
    "performance-review": {
        "name": "Performance Review Aggregator",
        "description": "Aggregate performance reviews and summarize key signals.",
    },

    # Finance
    "budget-actuals": {
        "name": "Budget vs Actuals Analyzer",
        "description": "Compare financial budgets against actual spending.",
    },
    "expense-auditor": {
        "name": "Expense Report Auditor",
        "description": "Audit expense reports and flag policy violations.",
    },
    "ar-aging": {
        "name": "AR Aging Analyst",
        "description": "Analyze accounts receivable aging and risk exposure.",
    },
    "cashflow-forecast": {
        "name": "Cash Flow Forecaster",
        "description": "Evaluate cash flow patterns and forecast liquidity.",
    },
    "payment-optimizer": {
        "name": "Payments Optimizer",
        "description": "Recommend payment prioritization and cash optimization.",
    },

    # Operations
    "supply-chain": {
        "name": "Supply Chain Analyst",
        "description": "Review supply chain data and identify bottlenecks.",
    },
    "inventory-manager": {
        "name": "Inventory Manager",
        "description": "Analyze inventory levels and turnover data.",
    },
    "operations-tracker": {
        "name": "Operations Tracker",
        "description": "Track operations performance and process efficiency.",
    },
    "quality-monitor": {
        "name": "Quality Monitor",
        "description": "Review quality metrics and flag operational problems.",
    },

    # Sales & Marketing
    "leads-analyzer": {
        "name": "Leads Analyzer",
        "description": "Evaluate lead quality and conversion potential.",
    },
    "campaign-performance": {
        "name": "Campaign Performance Analyst",
        "description": "Analyze campaign outcomes and identify winning tactics.",
    },
    "customer-journey": {
        "name": "Customer Journey Analyst",
        "description": "Map customer journeys and surface improvement areas.",
    },

    # IT & Compliance
    "security-audit": {
        "name": "Security Audit Analyst",
        "description": "Review compliance and security-related data.",
    },
    "compliance-monitor": {
        "name": "Compliance Monitor",
        "description": "Analyze compliance metrics and detect risks.",
    },
    "incident-response": {
        "name": "Incident Response Analyst",
        "description": "Help assess incident response and postmortem data.",
    },
    "it-asset-manager": {
        "name": "IT Asset Manager",
        "description": "Analyze IT asset data and identify optimization opportunities.",
    },

    # Data & Analytics
    "customer-segmentation": {
        "name": "Customer Segmentation Analyst",
        "description": "Segment customer data for better targeting.",
    },
    "sales-forecast": {
        "name": "Sales Forecast Analyst",
        "description": "Generate sales forecasts from historical data.",
    },
    "churn-predictor": {
        "name": "Churn Predictor",
        "description": "Analyze churn signals and retention opportunities.",
    },
    "pricing-optimizer": {
        "name": "Pricing Optimizer",
        "description": "Evaluate pricing scenarios and margin impact.",
    },
    "market-trends": {
        "name": "Market Trends Analyst",
        "description": "Review market data and highlight emerging trends.",
    },
    "data-agent": {
        "name": "Data Specialist",
        "description": "Read spreadsheets, compute aggregates, and cleanse data.",
    },
    "report-agent": {
        "name": "Report Generator",
        "description": "Generate Word/PDF reports from structured data.",
    },
    "communication-agent": {
        "name": "Communication Specialist",
        "description": "Send emails and notifications on behalf of workflows.",
    },
    "risk-agent": {
        "name": "Risk Specialist",
        "description": "Score loan applications and flag risky clients.",
    },
    "search-agent": {
        "name": "Search Specialist",
        "description": "RAG-based document search and retrieval.",
    },
    "csv-analyst": {
        "name": "CSV Data Analyst",
        "description": "Analyze uploaded CSV files with pandas and matplotlib.",
    },
    "sql-analyst": {
        "name": "SQL Database Analyst",
        "description": "Run read-only SQL against uploaded SQLite DBs or analyze table data.",
    },
    "excel-analyst": {
        "name": "Excel Multi-Sheet Analyst",
        "description": "Analyze Excel workbooks across sheets, summarize structure, and recommend pivots.",
    },
    "financial-data": {
        "name": "Financial Data Analyst",
        "description": "Analyze financial time series, compute technical indicators, and forecast price movements.",
    },
    "payroll-analyst": {
        "name": "Payroll Analyst",
        "description": "Analyze payroll data, flag anomalies, compute department costs, and detect compliance issues.",
    },
    "sales-pipeline": {
        "name": "Sales Pipeline Analyst",
        "description": "Analyze sales pipeline, compute win rates, forecast revenue, and identify bottlenecks.",
    },
    "json-analyst": {
        "name": "JSON & API Analyst",
        "description": "Analyze JSON data, flatten nested structures, compute statistics, and detect quality issues.",
    },
    "churn-analyzer": {
        "name": "Customer Churn Analyzer",
        "description": "Identify at-risk customers, compute churn rates, and recommend retention strategies.",
    },
    "attendance-analyzer": {
        "name": "Attendance Analyzer",
        "description": "Track attendance patterns, identify absenteeism, and compute utilization metrics.",
    },
    "vendor-spend": {
        "name": "Vendor Spend Analyzer",
        "description": "Analyze vendor spending patterns and consolidation opportunities.",
    },
    "project-timeline": {
        "name": "Project Timeline Analyzer",
        "description": "Analyze project schedules, milestone tracking, and timeline variance.",
    },
    "sla-compliance": {
        "name": "SLA Compliance Analyzer",
        "description": "Monitor and analyze Service Level Agreement compliance metrics.",
    },
    "inventory-analyst": {
        "name": "Inventory Analyst",
        "description": "Analyze inventory levels, stock movements, and supply chain metrics.",
    },
    "survey-analyzer": {
        "name": "Survey Analyzer",
        "description": "Analyze survey responses and customer feedback patterns.",
    },
    "access-rights": {
        "name": "Access Rights Analyzer",
        "description": "Analyze user access rights and security compliance.",
    },
    "license-tracker": {
        "name": "License Tracker",
        "description": "Track software licenses and monitor usage compliance.",
    },
    "incident-analyzer": {
        "name": "Incident Analyzer",
        "description": "Analyze IT incidents and support ticket metrics.",
    },
    "ml-modeler": {
        "name": "ML Modeler",
        "description": "Analyze data for machine learning model readiness and create basic predictive models.",
    },
    "log-analyst": {
        "name": "Log Analyst",
        "description": "Analyze system logs and application logs for patterns and anomalies.",
    },
    "image-processor": {
        "name": "Image Processor",
        "description": "Analyze images and metadata for image-based data intelligence.",
    },
    "pdf-extractor": {
        "name": "PDF Extractor",
        "description": "Extract and analyze data from PDF documents.",
    },
    "timeseries-forecaster": {
        "name": "Timeseries Forecaster",
        "description": "Analyze time series data and forecast trends.",
    },
    "multifile-correlation": {
        "name": "Multifile Correlation Analyzer",
        "description": "Analyze correlations across multiple data files and sources.",
    },
    "data-quality": {
        "name": "Data Quality Analyzer",
        "description": "Perform comprehensive data quality assessment and profiling.",
    },
}


def get_agent_info(agent_id: str) -> Optional[Dict[str, str]]:
    return AGENT_REGISTRY.get(agent_id)


def is_known_agent(agent_id: str) -> bool:
    return agent_id in AGENT_REGISTRY
