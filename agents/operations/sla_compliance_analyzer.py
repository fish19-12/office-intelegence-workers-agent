"""
SLA Compliance Analyzer Agent
Monitors and analyzes Service Level Agreement compliance.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_sla_compliance_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze SLA compliance and performance metrics.
    Expected input: CSV with ticket_id, service, sla_target, actual_time, status, priority columns
    or table_json with SLA records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("sla-compliance requires CSV or table_json input")

        result = {
            "total_tickets": len(df),
        }

        # SLA compliance calculation
        if "sla_target" in df.columns and "actual_time" in df.columns:
            compliant = (df["actual_time"] <= df["sla_target"]).sum()
            non_compliant = (df["actual_time"] > df["sla_target"]).sum()
            
            result["sla_compliance"] = {
                "compliant": compliant,
                "non_compliant": non_compliant,
                "compliance_rate_pct": (compliant / len(df) * 100) if len(df) > 0 else 0,
            }

        # Service analysis
        if "service" in df.columns:
            service_compliance = df.groupby("service").apply(
                lambda x: ((x["actual_time"] <= x["sla_target"]).sum() / len(x) * 100) if "sla_target" in x.columns and "actual_time" in x.columns else 0
            ).to_dict()
            result["by_service"] = service_compliance

        # Priority analysis
        if "priority" in df.columns:
            priority_dist = df["priority"].value_counts().to_dict()
            result["by_priority"] = priority_dist

        # Response time analysis
        if "actual_time" in df.columns:
            result["response_time_analysis"] = {
                "avg_response_time": df["actual_time"].mean(),
                "median_response_time": df["actual_time"].median(),
                "min_response_time": df["actual_time"].min(),
                "max_response_time": df["actual_time"].max(),
                "std_dev": df["actual_time"].std(),
            }

        # Status analysis
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist

        # Breach risk
        if "sla_target" in df.columns and "actual_time" in df.columns:
            at_risk = df[df["actual_time"] > (df["sla_target"] * 0.9)]
            result["breach_risk"] = {
                "tickets_at_risk": len(at_risk),
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "sla-compliance"}
