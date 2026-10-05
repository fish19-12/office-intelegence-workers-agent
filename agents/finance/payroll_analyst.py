from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


PAYROLL_PROMPT = """You are an elite HR payroll analyst focused on compensation accuracy and policy compliance.

MISSION
- Review payroll and compensation data for anomalies, inconsistencies, and policy issues.
- Compute department cost totals and compensation outliers.
- Identify possible overpayments, underpayments, and compliance risks.
- Recommend corrections and governance improvements grounded in the data.

OPERATING STANDARDS
- Treat every compensation exception as a possible data issue until verified.
- Distinguish between legitimate compensation differences and anomalies.
- Use precise metrics and thresholds when available.
- Present findings with actionable risk and remediation guidance.
"""


def analyze_payroll_data(csv_content: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Analyze payroll CSV with columns like: Employee, Department, Salary, Benefits, Deductions."""
    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_content))
        
        results: Dict[str, Any] = {
            "rows": len(df),
            "columns": df.columns.tolist(),
        }
        
        # Department-wise analysis if 'Department' column exists
        if 'Department' in df.columns and 'Salary' in df.columns:
            dept_analysis = {}
            for dept in df['Department'].unique():
                dept_df = df[df['Department'] == dept]
                salaries = pd.to_numeric(dept_df['Salary'], errors='coerce')
                dept_analysis[str(dept)] = {
                    "employee_count": len(dept_df),
                    "total_salary": float(salaries.sum()),
                    "avg_salary": float(salaries.mean()),
                    "salary_range": [float(salaries.min()), float(salaries.max())],
                }
            results["department_summary"] = dept_analysis
        
        # Flag anomalies
        if 'Salary' in df.columns:
            salaries = pd.to_numeric(df['Salary'], errors='coerce')
            q1, q3 = salaries.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = df[(salaries < lower_bound) | (salaries > upper_bound)]
            if len(anomalies) > 0:
                results["anomalies"] = {
                    "count": len(anomalies),
                    "rows": anomalies.index.tolist(),
                }
        
        return results
    except Exception as e:
        return {"error": str(e)}


def run_payroll_analyst(csv_content: Optional[str], table_json: Optional[List[Dict[str, Any]]], prompt: str) -> Dict[str, Any]:
    if csv_content:
        return analyze_payroll_data(csv_content, prompt)
    if table_json is not None:
        df = pd.DataFrame(table_json)
        return {"rows": len(df), "columns": df.columns.tolist(), "preview": df.head(3).to_dict(orient="records")}
    return {"error": "payroll-analyst requires CSV or table JSON input."}
