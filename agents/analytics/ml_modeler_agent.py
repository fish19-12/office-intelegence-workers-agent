"""
ML Modeler Agent
Analyzes data for machine learning model readiness and creates basic predictive models.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_ml_modeler(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze data for ML modeling and provide model recommendations.
    Expected input: CSV with feature columns and target variable
    or table_json with model training data.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("ml-modeler requires CSV or table_json input")

        result = {
            "total_records": len(df),
            "total_features": len(df.columns),
        }

        # Data quality assessment
        result["data_quality"] = {
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        }

        # Feature statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            result["numeric_feature_stats"] = {
                col: {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "skewness": df[col].skew(),
                }
                for col in numeric_cols
            }

        # Categorical feature analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            result["categorical_features"] = {
                col: df[col].nunique() for col in categorical_cols
            }

        # Class imbalance detection (if target variable exists)
        if len(df.columns) > 1:
            potential_target = df.iloc[:, -1]
            if potential_target.dtype == 'object' or potential_target.nunique() < 20:
                class_dist = potential_target.value_counts().to_dict()
                result["class_distribution"] = class_dist

        # Feature correlation (numeric only)
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr().values.tolist()
            result["feature_correlation_summary"] = {
                "highly_correlated_pairs": len([x for row in correlations for x in row if abs(x) > 0.8]) // 2,
            }

        # Model recommendations
        result["model_recommendations"] = {
            "data_readiness": "READY" if df.isnull().sum().sum() / (len(df) * len(df.columns)) < 0.1 else "NEEDS_CLEANING",
            "suggested_models": ["linear_regression", "random_forest", "gradient_boosting"] if len(numeric_cols) > 1 else ["classification_models"],
            "preprocessing_needed": bool(df.isnull().any().any()),
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "ml-modeler"}
