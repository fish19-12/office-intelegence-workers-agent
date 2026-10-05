"""
Inventory Analyst Agent
Analyzes inventory levels, stock movements, and supply chain metrics.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_inventory_analyst(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze inventory and stock management.
    Expected input: CSV with sku, product_name, qty_on_hand, qty_reserved, reorder_point, cost, value columns
    or table_json with inventory records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("inventory-analyst requires CSV or table_json input")

        result = {
            "total_skus": len(df),
        }

        # Inventory value
        if "value" in df.columns:
            result["total_inventory_value"] = df["value"].sum()
            result["avg_sku_value"] = df["value"].mean()

        # Stock level analysis
        if "qty_on_hand" in df.columns:
            result["stock_analysis"] = {
                "total_units": df["qty_on_hand"].sum(),
                "avg_units_per_sku": df["qty_on_hand"].mean(),
            }

        # Reserved stock
        if "qty_reserved" in df.columns and "qty_on_hand" in df.columns:
            result["reserved_stock"] = {
                "total_reserved": df["qty_reserved"].sum(),
                "available_after_reserved": (df["qty_on_hand"] - df["qty_reserved"]).sum(),
            }

        # Low stock warnings
        if "qty_on_hand" in df.columns and "reorder_point" in df.columns:
            low_stock = df[df["qty_on_hand"] <= df["reorder_point"]]
            result["low_stock_alerts"] = {
                "skus_below_reorder": len(low_stock),
                "total_units_below_reorder": low_stock["qty_on_hand"].sum() if len(low_stock) > 0 else 0,
            }

        # Product analysis
        if "product_name" in df.columns:
            product_dist = df["product_name"].value_counts().head(10).to_dict()
            result["top_products"] = product_dist

        # ABC analysis (by value)
        if "value" in df.columns:
            sorted_by_value = df.sort_values("value", ascending=False)
            total_value = sorted_by_value["value"].sum()
            cumulative_pct = sorted_by_value["value"].cumsum() / total_value * 100
            
            a_items = (cumulative_pct <= 80).sum()
            b_items = ((cumulative_pct > 80) & (cumulative_pct <= 95)).sum()
            c_items = (cumulative_pct > 95).sum()
            
            result["abc_analysis"] = {
                "a_items": a_items,
                "b_items": b_items,
                "c_items": c_items,
            }

        # Turnover analysis
        if "qty_on_hand" in df.columns and "cost" in df.columns:
            result["inventory_health"] = {
                "high_value_skus": (df["value"] > df["value"].quantile(0.75)).sum() if "value" in df.columns else 0,
                "slow_moving_items": (df["qty_on_hand"] > df["qty_on_hand"].quantile(0.75)).sum(),
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "inventory-analyst"}
