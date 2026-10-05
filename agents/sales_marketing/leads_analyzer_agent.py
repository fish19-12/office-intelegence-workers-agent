"""
Leads Analyzer Agent
Analyzes sales leads quality, conversion, and pipeline metrics.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_leads_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze sales leads and lead scoring.
    Expected input: CSV with lead_id, source, status, company, value, stage, age_days columns
    or table_json with lead records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("leads-analyzer requires CSV or table_json input")

        result = {
            "total_leads": len(df),
        }

        # Lead status distribution
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist
            result["qualified_leads"] = (df["status"] == "qualified").sum() if "status" in df.columns else 0
            result["converted_leads"] = (df["status"] == "converted").sum() if "status" in df.columns else 0

        # Lead value analysis
        if "value" in df.columns:
            result["value_analysis"] = {
                "total_pipeline_value": df["value"].sum(),
                "avg_lead_value": df["value"].mean(),
                "max_lead_value": df["value"].max(),
                "median_lead_value": df["value"].median(),
            }

        # Source analysis
        if "source" in df.columns:
            source_dist = df["source"].value_counts().to_dict()
            result["by_source"] = source_dist
            
            # Source efficiency
            if "status" in df.columns and "value" in df.columns:
                source_performance = df.groupby("source").apply(
                    lambda x: {
                        "converted": (x["status"] == "converted").sum() if "status" in x.columns else 0,
                        "avg_value": x["value"].mean() if "value" in x.columns else 0,
                    }
                ).to_dict()
                result["source_efficiency"] = source_performance

        # Pipeline stage analysis
        if "stage" in df.columns:
            stage_dist = df["stage"].value_counts().sort_index().to_dict()
            result["by_stage"] = stage_dist

        # Lead aging
        if "age_days" in df.columns:
            result["lead_aging"] = {
                "avg_age_days": df["age_days"].mean(),
                "leads_over_30_days": (df["age_days"] > 30).sum(),
                "leads_over_60_days": (df["age_days"] > 60).sum(),
                "leads_over_90_days": (df["age_days"] > 90).sum(),
            }

        # Conversion metrics
        if "status" in df.columns:
            converted = (df["status"] == "converted").sum()
            result["conversion_metrics"] = {
                "conversion_rate_pct": (converted / len(df) * 100) if len(df) > 0 else 0,
                "lead_quality_index": (converted / len(df) * 100) if len(df) > 0 else 0,
            }

        # Company analysis
        if "company" in df.columns:
            company_dist = df["company"].value_counts().head(10).to_dict()
            result["top_companies"] = company_dist

        return result

    except Exception as e:
        return {"error": str(e), "agent": "leads-analyzer"}
