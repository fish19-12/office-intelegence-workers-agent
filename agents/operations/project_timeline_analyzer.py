"""
Project Timeline Analyzer Agent
Analyzes project schedules, milestones, and timelines for delays and risks.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO
from datetime import datetime


def run_project_timeline_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze project timelines and milestone tracking.
    Expected input: CSV with project_id, milestone, planned_date, actual_date, status, owner columns
    or table_json with project records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("project-timeline requires CSV or table_json input")

        result = {
            "total_projects": df.get("project_id", pd.Series()).nunique() if "project_id" in df.columns else 0,
            "total_milestones": len(df),
        }

        # Project status
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist

        # Timeline variance
        if "planned_date" in df.columns and "actual_date" in df.columns:
            try:
                df["planned_date"] = pd.to_datetime(df["planned_date"])
                df["actual_date"] = pd.to_datetime(df["actual_date"])
                df_completed = df[df["actual_date"].notna()]
                
                if len(df_completed) > 0:
                    variance = (df_completed["actual_date"] - df_completed["planned_date"]).dt.days
                    on_time = (variance <= 0).sum()
                    delayed = (variance > 0).sum()
                    
                    result["timeline_analysis"] = {
                        "on_time_milestones": on_time,
                        "delayed_milestones": delayed,
                        "avg_delay_days": variance[variance > 0].mean() if len(variance[variance > 0]) > 0 else 0,
                        "max_delay_days": variance.max(),
                        "on_time_percentage": (on_time / len(df_completed) * 100) if len(df_completed) > 0 else 0,
                    }
            except:
                pass

        # Project owner analysis
        if "owner" in df.columns:
            owner_dist = df["owner"].value_counts().to_dict()
            result["by_owner"] = owner_dist

        # Risk assessment
        at_risk = df[df.get("status", pd.Series()).isin(["at_risk", "blocked"])]
        result["risk_assessment"] = {
            "at_risk_milestones": len(at_risk),
            "blocked_milestones": (df.get("status", pd.Series()) == "blocked").sum() if "status" in df.columns else 0,
        }

        # Upcoming milestones
        if "planned_date" in df.columns:
            try:
                df["planned_date"] = pd.to_datetime(df["planned_date"])
                today = pd.Timestamp.now()
                upcoming_within_30 = df[(df["planned_date"] >= today) & (df["planned_date"] <= today + pd.Timedelta(days=30))]
                result["upcoming_milestones_30_days"] = len(upcoming_within_30)
            except:
                pass

        return result

    except Exception as e:
        return {"error": str(e), "agent": "project-timeline"}
