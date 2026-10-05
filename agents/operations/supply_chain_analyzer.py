"""
Supply Chain Analyzer Agent
Analyzes supply chain efficiency, logistics, and vendor performance.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_supply_chain_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze supply chain and logistics metrics.
    Expected input: CSV with shipment_id, vendor, origin, destination, weight, cost, delivery_days, on_time columns
    or table_json with supply chain records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("supply-chain requires CSV or table_json input")

        result = {
            "total_shipments": len(df),
        }

        # Cost analysis
        if "cost" in df.columns:
            result["total_logistics_cost"] = df["cost"].sum()
            result["avg_shipment_cost"] = df["cost"].mean()

        # Delivery performance
        if "on_time" in df.columns:
            on_time_count = (df["on_time"] == True).sum() if "on_time" in df.columns else 0
            result["delivery_performance"] = {
                "on_time_deliveries": on_time_count,
                "late_deliveries": (df["on_time"] == False).sum() if "on_time" in df.columns else 0,
                "on_time_percentage": (on_time_count / len(df) * 100) if len(df) > 0 else 0,
            }

        # Delivery time analysis
        if "delivery_days" in df.columns:
            result["delivery_time_analysis"] = {
                "avg_delivery_days": df["delivery_days"].mean(),
                "median_delivery_days": df["delivery_days"].median(),
                "min_delivery_days": df["delivery_days"].min(),
                "max_delivery_days": df["delivery_days"].max(),
            }

        # Vendor performance
        if "vendor" in df.columns:
            vendor_performance = df.groupby("vendor").agg({
                "cost": ["sum", "mean"],
                "delivery_days": "mean",
                "on_time": lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0,
            }).to_dict("index")
            result["by_vendor"] = {k: v for k, v in list(vendor_performance.items())[:10]}

        # Route analysis
        if "origin" in df.columns and "destination" in df.columns:
            routes = df.groupby(["origin", "destination"]).size().to_dict()
            result["top_routes"] = dict(sorted(routes.items(), key=lambda x: x[1], reverse=True)[:10])

        # Weight analysis
        if "weight" in df.columns:
            result["weight_analysis"] = {
                "total_weight": df["weight"].sum(),
                "avg_weight_per_shipment": df["weight"].mean(),
            }

        # Cost efficiency
        if "weight" in df.columns and "cost" in df.columns:
            df["cost_per_weight"] = df["cost"] / (df["weight"] + 0.0001)
            result["cost_efficiency"] = {
                "avg_cost_per_unit_weight": df["cost_per_weight"].mean(),
                "most_efficient_routes": df.groupby(["origin", "destination"])["cost_per_weight"].mean().head(5).to_dict(),
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "supply-chain"}
