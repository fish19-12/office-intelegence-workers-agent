"""
Log Analyst Agent
Analyzes system logs and application logs for patterns and anomalies.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_log_analyst(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze system and application logs for issues and patterns.
    Expected input: CSV with timestamp, level, service, message, error_code columns
    or table_json with log records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("log-analyst requires CSV or table_json input")

        result = {
            "total_log_entries": len(df),
        }

        # Log level distribution
        if "level" in df.columns:
            level_dist = df["level"].value_counts().to_dict()
            result["log_level_distribution"] = level_dist
            result["error_count"] = (df["level"] == "ERROR").sum() if "level" in df.columns else 0
            result["warning_count"] = (df["level"] == "WARNING").sum() if "level" in df.columns else 0

        # Service analysis
        if "service" in df.columns:
            service_dist = df["service"].value_counts().to_dict()
            result["by_service"] = service_dist

        # Error code analysis
        if "error_code" in df.columns:
            error_dist = df["error_code"].value_counts().head(10).to_dict()
            result["top_errors"] = error_dist

        # Time-based analysis
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["hour"] = df["timestamp"].dt.hour
                hourly_dist = df["hour"].value_counts().sort_index().to_dict()
                result["logs_by_hour"] = hourly_dist
            except:
                pass

        # Error patterns
        if "level" in df.columns and "service" in df.columns:
            error_by_service = df[df["level"] == "ERROR"].groupby("service").size().to_dict()
            result["errors_by_service"] = error_by_service

        # Message analysis
        if "message" in df.columns:
            result["message_analysis"] = {
                "total_messages": df["message"].nunique(),
                "unique_message_types": df["message"].nunique(),
            }

        # Anomaly detection
        errors_df = df[df.get("level", pd.Series()) == "ERROR"]
        if len(errors_df) > 0:
            error_rate = (len(errors_df) / len(df) * 100)
            result["anomaly_detection"] = {
                "error_rate_pct": error_rate,
                "anomalous": "YES" if error_rate > 5 else "NO",
                "top_error": errors_df["error_code"].mode()[0] if len(errors_df["error_code"].mode()) > 0 and "error_code" in errors_df.columns else None,
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "log-analyst"}
