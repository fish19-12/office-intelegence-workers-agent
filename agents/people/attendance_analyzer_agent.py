from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


ATTENDANCE_PROMPT = """You are an elite HR attendance analyst focused on workforce reliability and policy compliance.

MISSION
- Analyze attendance patterns across teams and individuals.
- Identify chronic absenteeism, recurring lateness, and scheduling anomalies.
- Compute utilization and attendance rates using the underlying data.
- Flag policy violations and recommend corrective actions.

OPERATING STANDARDS
- Separate observed patterns from inferred explanations.
- Link findings to specific teams, roles, or time periods when possible.
- Use clear metrics such as attendance rate, absenteeism rate, and trend direction.
- Recommend operational improvements grounded in the observed data.
"""


def analyze_attendance_data(csv_content: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Analyze attendance CSV with columns: Employee, Department, Date, Status (Present/Absent/Leave)."""
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_content))
        
        results: Dict[str, Any] = {
            "rows": len(df),
            "columns": df.columns.tolist(),
        }
        
        # Status distribution
        if 'Status' in df.columns:
            status_dist = df['Status'].value_counts().to_dict()
            total = len(df)
            results["attendance_summary"] = {
                "total_records": total,
                "status_breakdown": status_dist,
                "attendance_rate_pct": round(100 * status_dist.get('Present', 0) / total, 2) if total > 0 else 0,
            }
        
        # Employee-level analysis
        if 'Employee' in df.columns and 'Status' in df.columns:
            employee_stats = {}
            for emp in df['Employee'].unique():
                emp_df = df[df['Employee'] == emp]
                present_count = (emp_df['Status'] == 'Present').sum()
                absent_count = (emp_df['Status'] == 'Absent').sum()
                employee_stats[str(emp)] = {
                    "records": len(emp_df),
                    "present": int(present_count),
                    "absent": int(absent_count),
                    "attendance_pct": round(100 * present_count / len(emp_df), 2) if len(emp_df) > 0 else 0,
                }
            results["employee_attendance"] = employee_stats
        
        # Department-level analysis
        if 'Department' in df.columns and 'Status' in df.columns:
            dept_stats = {}
            for dept in df['Department'].unique():
                dept_df = df[df['Department'] == dept]
                present = (dept_df['Status'] == 'Present').sum()
                dept_stats[str(dept)] = {
                    "records": len(dept_df),
                    "attendance_rate_pct": round(100 * present / len(dept_df), 2) if len(dept_df) > 0 else 0,
                }
            results["department_attendance"] = dept_stats
        
        return results
    except Exception as e:
        return {"error": str(e)}


def run_attendance_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict[str, Any]]], prompt: str) -> Dict[str, Any]:
    if csv_content:
        return analyze_attendance_data(csv_content, prompt)
    if table_json is not None:
        df = pd.DataFrame(table_json)
        return {"rows": len(df), "columns": df.columns.tolist()}
    return {"error": "attendance-analyzer requires CSV or table JSON input."}
