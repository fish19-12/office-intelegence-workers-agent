"""
Vendor Spend Analyzer Agent
Analyzes spending patterns across vendors and identifies optimization opportunities.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_vendor_spend_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze vendor spending and vendor consolidation opportunities.
    Expected input: CSV with vendor, amount, category, frequency, payment_terms columns
    or table_json with vendor spend records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("vendor-spend requires CSV or table_json input")

        result = {
            "total_vendors": df.get("vendor", pd.Series()).nunique() if "vendor" in df.columns else 0,
            "total_spend": df.get("amount", pd.Series()).sum() if "amount" in df.columns else 0,
        }

        # Vendor analysis
        if "vendor" in df.columns and "amount" in df.columns:
            vendor_spend = df.groupby("vendor")["amount"].agg(["sum", "count", "mean"]).reset_index()
            vendor_spend.columns = ["vendor", "total", "count", "avg"]
            vendor_spend_sorted = vendor_spend.sort_values("total", ascending=False)
            
            result["top_vendors"] = vendor_spend_sorted.head(10).to_dict("index")
            result["vendor_concentration"] = {
                "top_3_pct": (vendor_spend_sorted.head(3)["total"].sum() / result["total_spend"] * 100) if result["total_spend"] > 0 else 0,
                "top_10_pct": (vendor_spend_sorted.head(10)["total"].sum() / result["total_spend"] * 100) if result["total_spend"] > 0 else 0,
            }

        # Category analysis
        if "category" in df.columns:
            category_spend = df.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict()
            result["by_category"] = category_spend

        # Consolidation opportunities
        if "vendor" in df.columns and "category" in df.columns:
            category_vendor_count = df.groupby("category")["vendor"].nunique().to_dict()
            result["vendors_per_category"] = category_vendor_count

        # Payment terms analysis
        if "payment_terms" in df.columns:
            terms_dist = df["payment_terms"].value_counts().to_dict()
            result["payment_terms_distribution"] = terms_dist

        # Frequency analysis
        if "frequency" in df.columns:
            freq_dist = df["frequency"].value_counts().to_dict()
            result["purchase_frequency"] = freq_dist

        return result

    except Exception as e:
        return {"error": str(e), "agent": "vendor-spend"}
