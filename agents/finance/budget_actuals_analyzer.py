"""
Budget vs Actuals Analyzer Agent
Compares budgeted amounts to actual spending by department/category.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_budget_actuals_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze budget variance and actual vs planned spending.
    Expected input: CSV with department, category, budgeted, actual, period columns
    or table_json with budget records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("budget-actuals requires CSV or table_json input")

        result = {
            "total_budget": df.get("budgeted", pd.Series()).sum() if "budgeted" in df.columns else 0,
            "total_actual": df.get("actual", pd.Series()).sum() if "actual" in df.columns else 0,
        }

        # Calculate variance
        if "budgeted" in df.columns and "actual" in df.columns:
            df["variance"] = df["actual"] - df["budgeted"]
            df["variance_pct"] = ((df["actual"] - df["budgeted"]) / df["budgeted"] * 100).replace([float('inf'), float('-inf')], 0)
            
            result["total_variance"] = df["variance"].sum()
            result["variance_percentage"] = (result["total_variance"] / result["total_budget"] * 100) if result["total_budget"] != 0 else 0
            result["overspent_items"] = (df["variance"] > 0).sum()
            result["underspent_items"] = (df["variance"] < 0).sum()

            # Department analysis
            if "department" in df.columns:
                dept_variance = df.groupby("department").agg({
                    "budgeted": "sum",
                    "actual": "sum",
                    "variance": "sum"
                }).to_dict("index")
                result["by_department"] = {k: v for k, v in dept_variance.items()}

        # Category breakdown
        if "category" in df.columns:
            category_dist = df["category"].value_counts().head(10).to_dict()
            result["top_categories"] = category_dist

        return result

    except Exception as e:
        return {"error": str(e), "agent": "budget-actuals"}
