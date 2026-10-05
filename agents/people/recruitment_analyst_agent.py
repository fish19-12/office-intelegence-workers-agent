"""
Recruitment Analyst Agent
Analyzes recruitment pipeline, time-to-hire, and hiring efficiency metrics.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO
from datetime import datetime


def run_recruitment_analyst(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze recruitment metrics and hiring pipeline.
    Expected input: CSV with candidate_id, position, date_applied, date_hired, stage, source, salary_offered columns
    or table_json with recruitment records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("recruitment-analyst requires CSV or table_json input")

        result = {
            "total_candidates": len(df),
        }

        # Pipeline stage analysis
        if "stage" in df.columns:
            stage_dist = df["stage"].value_counts().to_dict()
            result["pipeline_by_stage"] = stage_dist
            result["hired_count"] = (df["stage"] == "hired").sum() if "stage" in df.columns else 0

        # Time-to-hire analysis
        if "date_applied" in df.columns and "date_hired" in df.columns:
            try:
                df["date_applied"] = pd.to_datetime(df["date_applied"])
                df["date_hired"] = pd.to_datetime(df["date_hired"])
                hired_df = df[df["date_hired"].notna()]
                if len(hired_df) > 0:
                    time_to_hire = (hired_df["date_hired"] - hired_df["date_applied"]).dt.days
                    result["time_to_hire"] = {
                        "avg_days": time_to_hire.mean(),
                        "median_days": time_to_hire.median(),
                        "min_days": time_to_hire.min(),
                        "max_days": time_to_hire.max(),
                    }
            except:
                pass

        # Position analysis
        if "position" in df.columns:
            position_dist = df["position"].value_counts().to_dict()
            result["by_position"] = position_dist

        # Source analysis
        if "source" in df.columns:
            source_dist = df["source"].value_counts().to_dict()
            result["candidates_by_source"] = source_dist

        # Salary analysis
        if "salary_offered" in df.columns:
            result["salary_analysis"] = {
                "avg_salary_offered": df["salary_offered"].mean(),
                "median_salary_offered": df["salary_offered"].median(),
                "salary_range": {
                    "min": df["salary_offered"].min(),
                    "max": df["salary_offered"].max(),
                },
            }

        # Conversion metrics
        hired = len(df[df.get("stage", pd.Series()) == "hired"]) if "stage" in df.columns else 0
        result["conversion_metrics"] = {
            "conversion_rate_pct": (hired / len(df) * 100) if len(df) > 0 else 0,
            "rejection_rate_pct": ((df.get("stage", pd.Series()) == "rejected").sum() / len(df) * 100) if len(df) > 0 else 0,
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "recruitment-analyst"}
