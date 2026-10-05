"""
Cashflow Forecast Agent
Forecasts cash flow based on historical trends and planned transactions.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_cashflow_forecast_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Forecast cash flow patterns and trends.
    Expected input: CSV with date, transaction_type, amount, category, forecast_period columns
    or table_json with transaction records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("cashflow-forecast requires CSV or table_json input")

        result = {
            "total_transactions": len(df),
        }

        # Inflow and outflow analysis
        if "transaction_type" in df.columns and "amount" in df.columns:
            inflows = df[df["transaction_type"] == "inflow"]["amount"].sum()
            outflows = df[df["transaction_type"] == "outflow"]["amount"].sum()
            result["total_inflows"] = inflows
            result["total_outflows"] = outflows
            result["net_cashflow"] = inflows - outflows
            result["net_cashflow_pct"] = ((inflows - outflows) / inflows * 100) if inflows > 0 else 0

        # Category breakdown
        if "category" in df.columns:
            category_dist = df.groupby("category")["amount"].agg(["sum", "count"]).to_dict("index")
            result["by_category"] = {k: v for k, v in category_dist.items()}

        # Forecast analysis
        if "forecast_period" in df.columns:
            forecast_dist = df["forecast_period"].value_counts().to_dict()
            result["forecast_periods"] = forecast_dist

        # Trend analysis
        if "date" in df.columns and "amount" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                monthly_total = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
                result["monthly_trend"] = monthly_total.to_dict()
            except:
                pass

        # Volatility assessment
        if "amount" in df.columns:
            result["volatility_metrics"] = {
                "std_dev": df["amount"].std(),
                "cv": df["amount"].std() / (df["amount"].mean() + 0.0001),  # Coefficient of variation
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "cashflow-forecast"}
