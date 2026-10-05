"""
Expense Auditor Agent
Audits expense reports for compliance, duplicates, and policy violations.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_expense_auditor(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Audit expense reports for compliance and anomalies.
    Expected input: CSV with expense_id, employee, amount, category, description, date, status columns
    or table_json with expense records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("expense-auditor requires CSV or table_json input")

        result = {
            "total_expenses": len(df),
            "total_amount": df.get("amount", pd.Series()).sum() if "amount" in df.columns else 0,
        }

        # Status analysis
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist
            result["approved_count"] = (df["status"] == "approved").sum() if "status" in df.columns else 0
            result["pending_count"] = (df["status"] == "pending").sum() if "status" in df.columns else 0
            result["rejected_count"] = (df["status"] == "rejected").sum() if "status" in df.columns else 0

        # Amount analysis for anomalies
        if "amount" in df.columns:
            q3 = df["amount"].quantile(0.75)
            q1 = df["amount"].quantile(0.25)
            iqr = q3 - q1
            outliers = df[(df["amount"] > q3 + 1.5 * iqr) | (df["amount"] < q1 - 1.5 * iqr)]
            
            result["amount_analysis"] = {
                "avg_expense": df["amount"].mean(),
                "max_expense": df["amount"].max(),
                "anomalous_expenses": len(outliers),
            }

        # Employee spending analysis
        if "employee" in df.columns:
            emp_dist = df["employee"].value_counts().head(10).to_dict()
            result["top_spenders"] = emp_dist

        # Category breakdown
        if "category" in df.columns:
            category_breakdown = df.groupby("category")["amount"].agg(["sum", "count"]).to_dict("index")
            result["by_category"] = {k: v for k, v in category_breakdown.items()}

        # Duplicate detection
        if "description" in df.columns:
            duplicate_descriptions = df[df.duplicated(subset=["description"], keep=False)]
            result["potential_duplicate_expenses"] = len(duplicate_descriptions)

        return result

    except Exception as e:
        return {"error": str(e), "agent": "expense-auditor"}
