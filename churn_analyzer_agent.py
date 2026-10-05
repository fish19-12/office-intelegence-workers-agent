from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


CHURN_PROMPT = """You are an elite customer churn analyst focused on risk detection and retention strategy.

MISSION
- Identify at-risk customers using customer behavior, lifecycle stage, and engagement patterns.
- Compute churn and retention metrics by cohort, segment, and time window.
- Highlight the biggest retention opportunities and likely drivers of churn.

OPERATING STANDARDS
- Use evidence from the dataset rather than generic churn theory.
- Prioritize the highest-risk segments and the largest revenue exposure.
- Recommend targeted retention interventions that fit each segment.
- Clearly distinguish between correlation and causal interpretation.
"""


def analyze_churn_data(csv_content: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Analyze churn CSV with columns: Customer, Signup_Date, Churn_Date, Status, Revenue."""
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_content))
        
        results: Dict[str, Any] = {
            "rows": len(df),
            "columns": df.columns.tolist(),
        }
        
        # Churn rate
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts().to_dict()
            total = len(df)
            churn_count = status_counts.get('churned', status_counts.get('Churned', 0))
            results["churn_metrics"] = {
                "total_customers": total,
                "churned": int(churn_count),
                "churn_rate_pct": round(100 * churn_count / total, 2) if total > 0 else 0,
                "status_breakdown": status_counts,
            }
        
        # Revenue impact
        if 'Revenue' in df.columns and 'Status' in df.columns:
            revenue_by_status = df.groupby('Status')['Revenue'].agg(['sum', 'mean', 'count']).to_dict('index')
            results["revenue_analysis"] = {str(k): {str(k2): float(v2) if isinstance(v2, (int, float)) else v2 for k2, v2 in v.items()} for k, v in revenue_by_status.items()}
        
        return results
    except Exception as e:
        return {"error": str(e)}


def run_churn_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict[str, Any]]], prompt: str) -> Dict[str, Any]:
    if csv_content:
        return analyze_churn_data(csv_content, prompt)
    if table_json is not None:
        df = pd.DataFrame(table_json)
        return {"rows": len(df), "columns": df.columns.tolist()}
    return {"error": "churn-analyzer requires CSV or table JSON input."}
