"""
Multifile Correlation Agent
Analyzes correlations across multiple data files and sources.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_multifile_correlation_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze correlations across multiple files or datasets.
    Expected input: CSV with file_id, metric_name, value, timestamp columns
    or table_json with multi-file correlation data.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("multifile-correlation requires CSV or table_json input")

        result = {
            "total_records": len(df),
        }

        # File analysis
        if "file_id" in df.columns:
            file_count = df["file_id"].nunique()
            result["unique_files"] = file_count
            file_dist = df["file_id"].value_counts().to_dict()
            result["records_per_file"] = file_dist

        # Metric analysis
        if "metric_name" in df.columns:
            metric_dist = df["metric_name"].value_counts().to_dict()
            result["by_metric"] = metric_dist

        # Cross-file correlation
        if "file_id" in df.columns and "metric_name" in df.columns and "value" in df.columns:
            # Pivot data for correlation analysis
            pivot_df = df.pivot_table(
                index="file_id",
                columns="metric_name",
                values="value",
                aggfunc="mean"
            )
            
            if len(pivot_df.columns) > 1:
                correlations = pivot_df.corr()
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(correlations.columns)):
                    for j in range(i+1, len(correlations.columns)):
                        corr_val = correlations.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append({
                                "metric1": correlations.columns[i],
                                "metric2": correlations.columns[j],
                                "correlation": float(corr_val),
                            })
                
                result["strong_correlations"] = strong_corr

        # Temporal analysis
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                date_range = (df["timestamp"].max() - df["timestamp"].min()).days
                result["temporal_analysis"] = {
                    "date_range_days": date_range,
                    "time_coverage": "COMPLETE" if date_range > 30 else "PARTIAL",
                }
            except:
                pass

        # Value distribution
        if "value" in df.columns:
            result["value_distribution"] = {
                "mean": df["value"].mean(),
                "std": df["value"].std(),
                "min": df["value"].min(),
                "max": df["value"].max(),
            }

        # Data completeness
        result["data_completeness"] = {
            "total_nulls": df.isnull().sum().sum(),
            "null_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "multifile-correlation"}
