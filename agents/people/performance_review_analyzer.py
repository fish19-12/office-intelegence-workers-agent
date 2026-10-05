"""
Performance Review Agent
Analyzes employee performance data and generates performance insights.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_performance_review_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze employee performance reviews and metrics.
    Expected input: CSV with employee_id, department, rating, review_score, goals_achieved, review_date columns
    or table_json with performance records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("performance-review requires CSV or table_json input")

        result = {
            "total_reviews": len(df),
        }

        # Rating distribution
        if "rating" in df.columns:
            rating_dist = df["rating"].value_counts().sort_index().to_dict()
            result["rating_distribution"] = rating_dist
            result["avg_rating"] = df["rating"].mean()

        # Score analysis
        if "review_score" in df.columns:
            result["score_analysis"] = {
                "avg_review_score": df["review_score"].mean(),
                "median_review_score": df["review_score"].median(),
                "min_review_score": df["review_score"].min(),
                "max_review_score": df["review_score"].max(),
                "std_dev": df["review_score"].std(),
            }

        # Goal achievement analysis
        if "goals_achieved" in df.columns:
            result["goal_achievement"] = {
                "total_goals_achieved": df["goals_achieved"].sum(),
                "avg_goals_per_employee": df["goals_achieved"].mean(),
                "employees_with_all_goals": (df["goals_achieved"] >= df["goals_achieved"].max()).sum() if len(df) > 0 else 0,
            }

        # Department analysis
        if "department" in df.columns:
            dept_stats = df.groupby("department").agg({
                "review_score": ["mean", "count"],
                "rating": lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            }).to_dict("index")
            result["by_department"] = {k: v for k, v in dept_stats.items()}

        # Performance tiers
        if "review_score" in df.columns:
            high_performers = (df["review_score"] >= df["review_score"].quantile(0.75)).sum()
            mid_performers = (df["review_score"] >= df["review_score"].quantile(0.25)).sum() - high_performers
            low_performers = len(df) - high_performers - mid_performers
            
            result["performance_tiers"] = {
                "high_performers": high_performers,
                "mid_performers": mid_performers,
                "low_performers": low_performers,
            }

        # Improvement areas
        if "review_score" in df.columns:
            low_score_employees = df[df["review_score"] < df["review_score"].quantile(0.25)]
            result["improvement_opportunities"] = {
                "employees_needing_support": len(low_score_employees),
                "avg_score_of_low_performers": low_score_employees["review_score"].mean() if len(low_score_employees) > 0 else 0,
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "performance-review"}
