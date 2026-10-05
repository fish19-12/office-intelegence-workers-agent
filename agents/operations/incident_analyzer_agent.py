"""
Incident Analyzer Agent
Analyzes IT incidents and support tickets.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_incident_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze IT incidents and support metrics.
    Expected input: CSV with incident_id, severity, status, category, time_to_resolve, assigned_to, reported_date columns
    or table_json with incident records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("incident-analyzer requires CSV or table_json input")

        result = {
            "total_incidents": len(df),
        }

        # Severity distribution
        if "severity" in df.columns:
            severity_dist = df["severity"].value_counts().to_dict()
            result["severity_distribution"] = severity_dist
            result["critical_incidents"] = (df["severity"] == "critical").sum() if "severity" in df.columns else 0
            result["high_incidents"] = (df["severity"] == "high").sum() if "severity" in df.columns else 0

        # Status distribution
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist
            result["open_incidents"] = (df["status"].isin(["open", "in_progress"])).sum() if "status" in df.columns else 0
            result["resolved_incidents"] = (df["status"] == "resolved").sum() if "status" in df.columns else 0

        # Category analysis
        if "category" in df.columns:
            category_dist = df["category"].value_counts().to_dict()
            result["by_category"] = category_dist

        # Resolution time analysis
        if "time_to_resolve" in df.columns:
            resolved_df = df[df["time_to_resolve"].notna()]
            if len(resolved_df) > 0:
                result["resolution_metrics"] = {
                    "avg_time_to_resolve_hours": resolved_df["time_to_resolve"].mean(),
                    "median_time_to_resolve_hours": resolved_df["time_to_resolve"].median(),
                    "max_time_to_resolve_hours": resolved_df["time_to_resolve"].max(),
                    "min_time_to_resolve_hours": resolved_df["time_to_resolve"].min(),
                }

        # Technician assignment
        if "assigned_to" in df.columns:
            assigned_dist = df["assigned_to"].value_counts().to_dict()
            result["by_technician"] = assigned_dist

        # Incident trend
        if "reported_date" in df.columns:
            try:
                df["reported_date"] = pd.to_datetime(df["reported_date"])
                daily_incidents = df.groupby(df["reported_date"].dt.date).size().to_dict()
                result["incident_trend"] = daily_incidents
            except:
                pass

        # Severity vs time analysis
        if "severity" in df.columns and "time_to_resolve" in df.columns:
            severity_time = df.groupby("severity")["time_to_resolve"].mean().to_dict()
            result["avg_resolution_time_by_severity"] = severity_time

        # MTTR (Mean Time To Resolve) by category
        if "category" in df.columns and "time_to_resolve" in df.columns:
            category_time = df.groupby("category")["time_to_resolve"].mean().to_dict()
            result["mttr_by_category"] = category_time

        # Incident health
        critical_unresolved = df[(df.get("severity", pd.Series()) == "critical") & (df.get("status", pd.Series()) != "resolved")]
        result["incident_health"] = {
            "critical_unresolved": len(critical_unresolved),
            "escalation_rate": ((df.get("severity", pd.Series()) == "critical").sum() / len(df) * 100) if len(df) > 0 else 0,
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "incident-analyzer"}
