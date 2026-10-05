from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


EXCEL_PROMPT = """You are an elite Excel analyst for multi-sheet workbooks.

MISSION
- Review the workbook as a structured dataset, not as isolated cells.
- Summarize each sheet's schema, size, quality, and likely analytical value.
- Highlight patterns, missing data, and opportunities for pivoting or cleaning.

OPERATING PROTOCOL
1. Enumerate all sheets and understand their purpose.
2. Inspect row counts, column counts, dtypes, and missingness per sheet.
3. Identify data quality issues: blanks, inconsistent headers, duplicated rows, and mismatched types.
4. Recommend the most useful summarization, pivot, or reporting approach.
5. Translate technical findings into clear operational guidance.

OUTPUT STANDARDS
- Provide a workbook summary with per-sheet details.
- Call out high-value insights and risks.
- Recommend where to clean, aggregate, or pivot data.

QUALITY GATES
- Be explicit when a sheet is too sparse or ambiguous for strong conclusions.
- Use the workbook structure and sample rows as evidence before giving recommendations.
"""


def _load_workbook(file_path: str) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    if not Path(file_path).suffix.lower() in {".xlsx", ".xls"}:
        raise ValueError("Excel analyst requires an .xlsx or .xls file.")
    return pd.read_excel(file_path, sheet_name=None)


def analyze_excel_workbook(file_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    workbook = _load_workbook(file_path)
    summary: List[Dict[str, Any]] = []
    for sheet_name, df in workbook.items():
        summary.append({
            "sheet": sheet_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "preview": df.head(5).to_dict(orient="records"),
        })

    return {
        "workbook_summary": {
            "sheet_count": len(workbook),
            "sheet_names": list(workbook.keys()),
            "sheets": summary,
        },
        "recommendation": _recommend_workbook_insights(workbook, prompt),
    }


def analyze_table_json(table_json: List[Dict[str, Any]], prompt: Optional[str] = None) -> Dict[str, Any]:
    df = pd.DataFrame(table_json)
    return {
        "sheet": "table_json",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "preview": df.head(5).to_dict(orient="records"),
        "recommendation": _recommend_table_insights(df, prompt),
    }


def _recommend_workbook_insights(workbook: Dict[str, pd.DataFrame], prompt: Optional[str] = None) -> str:
    insights = []
    for sheet_name, df in workbook.items():
        if len(df.columns) > 0:
            insights.append(f"Sheet '{sheet_name}' has {len(df)} rows and {len(df.columns)} columns.")
        if df.isna().sum().sum() > 0:
            insights.append(f"Sheet '{sheet_name}' contains missing values and may need cleaning.")
    if not insights:
        return "Workbook appears clean and well-structured."
    return " ".join(insights)


def _recommend_table_insights(df: pd.DataFrame, prompt: Optional[str] = None) -> str:
    if df.empty:
        return "Table JSON is empty."
    missing = df.isna().sum().sum()
    if missing > 0:
        return f"The table contains {missing} missing values; consider cleaning or imputing these values."
    return "The table looks complete and can be analyzed with pivot tables or summaries."


def run_excel_analyst(file_path: Optional[str], table_json: Optional[List[Dict[str, Any]]], prompt: str) -> Dict[str, Any]:
    if file_path:
        return analyze_excel_workbook(file_path, prompt)
    if table_json is not None:
        return analyze_table_json(table_json, prompt)
    raise ValueError("excel-analyst requires an Excel file path or table JSON input.")
