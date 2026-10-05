"""
Invoice Processor Agent
Analyzes invoice data for payment status, amounts, aging, and financial metrics.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO
from datetime import datetime, timedelta


def run_invoice_processor(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Process and analyze invoice records for financial tracking and aging.
    Expected input: CSV with invoice_id, vendor, amount, date_issued, date_due, status, amount_paid columns
    or table_json with invoice records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("invoice-processor requires CSV or table_json input")

        result = {
            "total_invoices": len(df),
            "total_amount": df.get("amount", pd.Series()).sum() if "amount" in df.columns else 0,
        }

        # Payment status distribution
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist
            result["pending_invoices"] = (df["status"] == "pending").sum() if "status" in df.columns else 0
            result["overdue_invoices"] = (df["status"] == "overdue").sum() if "status" in df.columns else 0

        # Amount analysis
        if "amount" in df.columns:
            result["amount_analysis"] = {
                "avg_invoice_amount": df["amount"].mean(),
                "max_invoice_amount": df["amount"].max(),
                "min_invoice_amount": df["amount"].min(),
            }

        # Payment tracking
        if "amount_paid" in df.columns and "amount" in df.columns:
            total_paid = df.get("amount_paid", pd.Series()).sum()
            total_due = df.get("amount", pd.Series()).sum()
            result["payment_tracking"] = {
                "total_paid": total_paid,
                "total_due": total_due,
                "outstanding_amount": total_due - total_paid,
                "payment_percentage": (total_paid / total_due * 100) if total_due > 0 else 0,
            }

        # Vendor analysis
        if "vendor" in df.columns:
            vendor_dist = df["vendor"].value_counts().head(10).to_dict()
            result["top_vendors"] = vendor_dist

        # Invoice aging
        if "date_due" in df.columns:
            try:
                df["date_due"] = pd.to_datetime(df["date_due"])
                today = datetime.now()
                overdue_days = (today - df["date_due"]).dt.days
                result["aging_analysis"] = {
                    "avg_days_overdue": overdue_days[overdue_days > 0].mean() if len(overdue_days[overdue_days > 0]) > 0 else 0,
                    "invoices_over_30_days": (overdue_days > 30).sum(),
                    "invoices_over_60_days": (overdue_days > 60).sum(),
                    "invoices_over_90_days": (overdue_days > 90).sum(),
                }
            except:
                pass

        return result

    except Exception as e:
        return {"error": str(e), "agent": "invoice-processor"}
