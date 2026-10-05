"""
Campaign Performance Analyzer Agent
Analyzes marketing campaign metrics and ROI.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_campaign_performance_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze marketing campaign performance and ROI.
    Expected input: CSV with campaign_id, channel, spent, impressions, clicks, conversions, revenue columns
    or table_json with campaign records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("campaign-performance requires CSV or table_json input")

        result = {
            "total_campaigns": len(df),
        }

        # Budget analysis
        if "spent" in df.columns:
            result["total_spend"] = df["spent"].sum()
            result["avg_spend_per_campaign"] = df["spent"].mean()

        # Engagement metrics
        if "impressions" in df.columns:
            result["total_impressions"] = df["impressions"].sum()
        
        if "clicks" in df.columns:
            result["total_clicks"] = df["clicks"].sum()

        # CTR calculation
        if "clicks" in df.columns and "impressions" in df.columns:
            result["click_through_rate_pct"] = (df["clicks"].sum() / (df["impressions"].sum() + 0.0001) * 100)

        # Conversion analysis
        if "conversions" in df.columns:
            result["total_conversions"] = df["conversions"].sum()
            if "clicks" in df.columns:
                result["conversion_rate_pct"] = (df["conversions"].sum() / (df["clicks"].sum() + 0.0001) * 100)

        # ROI calculation
        if "spent" in df.columns and "revenue" in df.columns:
            total_revenue = df["revenue"].sum()
            total_spent = df["spent"].sum()
            result["roi_analysis"] = {
                "total_revenue": total_revenue,
                "total_investment": total_spent,
                "net_profit": total_revenue - total_spent,
                "roi_percentage": ((total_revenue - total_spent) / (total_spent + 0.0001) * 100),
            }

        # Channel analysis
        if "channel" in df.columns:
            channel_performance = df.groupby("channel").agg({
                "spent": "sum",
                "conversions": "sum",
                "revenue": "sum",
            }).to_dict("index")
            result["by_channel"] = {k: v for k, v in channel_performance.items()}

        # Cost per acquisition
        if "spent" in df.columns and "conversions" in df.columns:
            result["cpa_analysis"] = {
                "avg_cost_per_acquisition": (df["spent"].sum() / (df["conversions"].sum() + 0.0001)),
            }

        # Performance ranking
        if "roi_percentage" in locals() or ("spent" in df.columns and "revenue" in df.columns):
            df_copy = df.copy()
            if "spent" in df_copy.columns and "revenue" in df_copy.columns:
                df_copy["roi"] = (df_copy["revenue"] - df_copy["spent"]) / (df_copy["spent"] + 0.0001)
                top_performers = df_copy.nlargest(5, "roi")[["campaign_id", "roi"]] if "campaign_id" in df_copy.columns else df_copy.nlargest(5, "roi")
                result["top_performing_campaigns"] = top_performers.to_dict("index") if len(top_performers) > 0 else {}

        return result

    except Exception as e:
        return {"error": str(e), "agent": "campaign-performance"}
