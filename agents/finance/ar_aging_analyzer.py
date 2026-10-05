"""
AR Aging Analyzer Agent
Analyzes Accounts Receivable aging and collection patterns.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO
from datetime import datetime, timedelta


def run_ar_aging_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze receivables aging and collection efficiency.
    Expected input: CSV with invoice_id, customer, amount, date_issued, date_due, amount_paid, days_outstanding columns
    or table_json with AR records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("ar-aging requires CSV or table_json input")

        result = {
            "total_outstanding": df.get("amount", pd.Series()).sum() if "amount" in df.columns else 0,
            "total_paid": df.get("amount_paid", pd.Series()).sum() if "amount_paid" in df.columns else 0,
        }

        # Calculate collection rate
        if "amount" in df.columns and "amount_paid" in df.columns:
            total_amount = df["amount"].sum()
            total_paid = df["amount_paid"].sum()
            result["collection_rate_pct"] = (total_paid / total_amount * 100) if total_amount > 0 else 0

        # Aging bucket analysis
        if "days_outstanding" in df.columns:
            result["aging_buckets"] = {
                "current": (df["days_outstanding"] <= 30).sum(),
                "30_60": ((df["days_outstanding"] > 30) & (df["days_outstanding"] <= 60)).sum(),
                "60_90": ((df["days_outstanding"] > 60) & (df["days_outstanding"] <= 90)).sum(),
                "over_90": (df["days_outstanding"] > 90).sum(),
            }
            result["avg_days_outstanding"] = df["days_outstanding"].mean()

        # Customer analysis
        if "customer" in df.columns:
            customer_analysis = df.groupby("customer").agg({
                "amount": "sum",
                "amount_paid": "sum"
            }).to_dict("index")
            result["top_customers"] = {k: v for k, v in list(customer_analysis.items())[:10]}

        # Collection efficiency
        overdue_invoices = df[df.get("days_outstanding", pd.Series()) > 30]
        result["collection_metrics"] = {
            "invoices_over_30_days": len(overdue_invoices),
            "amount_over_30_days": overdue_invoices.get("amount", pd.Series()).sum() if len(overdue_invoices) > 0 else 0,
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "ar-aging"}
