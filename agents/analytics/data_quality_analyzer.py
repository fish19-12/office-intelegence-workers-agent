"""
Data Quality Agent
Performs comprehensive data quality assessment and profiling.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_data_quality_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Perform comprehensive data quality assessment.
    Expected input: CSV with any columns for quality assessment
    or table_json with data records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("data-quality requires CSV or table_json input")

        result = {
            "total_records": len(df),
            "total_columns": len(df.columns),
        }

        # Missing data assessment
        missing_analysis = df.isnull().sum()
        result["missing_data"] = {
            "total_missing": missing_analysis.sum(),
            "missing_percentage": (missing_analysis.sum() / (len(df) * len(df.columns)) * 100),
            "by_column": {col: int(count) for col, count in missing_analysis.items() if count > 0},
        }

        # Duplicate analysis
        duplicates = df.duplicated().sum()
        result["duplicates"] = {
            "total_duplicate_rows": duplicates,
            "duplicate_percentage": (duplicates / len(df) * 100) if len(df) > 0 else 0,
        }

        # Data type analysis
        result["data_types"] = {
            "numeric": len(df.select_dtypes(include=['number']).columns),
            "string": len(df.select_dtypes(include=['object']).columns),
            "datetime": len(df.select_dtypes(include=['datetime64']).columns),
            "other": len(df.select_dtypes(exclude=['number', 'object', 'datetime64']).columns),
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result["numeric_analysis"] = {}
            for col in numeric_cols:
                col_data = df[col].dropna()
                result["numeric_analysis"][col] = {
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "outliers": len(col_data[(col_data < col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))) | (col_data > col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))]),
                }

        # String columns analysis
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            result["string_analysis"] = {
                col: {
                    "unique_values": df[col].nunique(),
                    "empty_strings": (df[col] == "").sum(),
                    "most_common": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                }
                for col in string_cols
            }

        # Consistency checks
        result["consistency_checks"] = {
            "consistent_lengths": all(len(df) == len(col_data) + df[col].isnull().sum() for col, col_data in df.items()),
            "no_mixed_types": all(df[col].dtype != 'object' or df[col].apply(type).nunique() <= 2 for col in df.columns),
        }

        # Overall data quality score
        quality_score = 100
        if result["missing_data"]["missing_percentage"] > 10:
            quality_score -= 20
        if result["duplicates"]["duplicate_percentage"] > 5:
            quality_score -= 15
        if result["consistency_checks"]["consistent_lengths"] == False:
            quality_score -= 15

        result["overall_quality_score"] = max(quality_score, 0)
        result["quality_status"] = "EXCELLENT" if quality_score >= 85 else "GOOD" if quality_score >= 70 else "FAIR" if quality_score >= 50 else "POOR"

        return result

    except Exception as e:
        return {"error": str(e), "agent": "data-quality"}
