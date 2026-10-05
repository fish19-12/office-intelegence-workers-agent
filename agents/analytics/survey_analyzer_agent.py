"""
Survey Analyzer Agent
Analyzes survey responses and customer feedback.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_survey_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze survey and feedback data.
    Expected input: CSV with response_id, question, rating, sentiment, category, respondent_type columns
    or table_json with survey records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("survey-analyzer requires CSV or table_json input")

        result = {
            "total_responses": len(df),
        }

        # Rating analysis
        if "rating" in df.columns:
            result["rating_distribution"] = df["rating"].value_counts().sort_index().to_dict()
            result["avg_rating"] = df["rating"].mean()
            result["median_rating"] = df["rating"].median()
            
            # NPS (Net Promoter Score) if scale is 0-10
            if df["rating"].max() <= 10:
                promoters = (df["rating"] >= 9).sum()
                detractors = (df["rating"] <= 6).sum()
                nps = ((promoters - detractors) / len(df) * 100) if len(df) > 0 else 0
                result["nps_score"] = nps

        # Sentiment analysis
        if "sentiment" in df.columns:
            sentiment_dist = df["sentiment"].value_counts().to_dict()
            result["sentiment_distribution"] = sentiment_dist

        # Category analysis
        if "category" in df.columns:
            category_dist = df["category"].value_counts().to_dict()
            result["by_category"] = category_dist

            # Category satisfaction
            if "rating" in df.columns:
                category_satisfaction = df.groupby("category")["rating"].agg(["mean", "count"]).to_dict("index")
                result["satisfaction_by_category"] = {k: v for k, v in category_satisfaction.items()}

        # Respondent type analysis
        if "respondent_type" in df.columns:
            respondent_dist = df["respondent_type"].value_counts().to_dict()
            result["by_respondent_type"] = respondent_dist

            # Satisfaction by respondent type
            if "rating" in df.columns:
                respondent_satisfaction = df.groupby("respondent_type")["rating"].agg(["mean", "count"]).to_dict("index")
                result["satisfaction_by_respondent"] = {k: v for k, v in respondent_satisfaction.items()}

        # Score distribution
        if "rating" in df.columns:
            high_score = (df["rating"] >= 4).sum() if df["rating"].max() <= 5 else (df["rating"] >= 8).sum()
            mid_score = ((df["rating"] >= 3) & (df["rating"] < 4)).sum() if df["rating"].max() <= 5 else ((df["rating"] >= 6) & (df["rating"] < 8)).sum()
            low_score = (df["rating"] < 3).sum() if df["rating"].max() <= 5 else (df["rating"] < 6).sum()
            
            result["satisfaction_tiers"] = {
                "satisfied": high_score,
                "neutral": mid_score,
                "dissatisfied": low_score,
            }

        # Improvement areas
        if "sentiment" in df.columns:
            negative_responses = df[df["sentiment"].isin(["negative", "dissatisfied"])]
            result["improvement_opportunities"] = {
                "negative_feedback_count": len(negative_responses),
                "pct_negative": (len(negative_responses) / len(df) * 100) if len(df) > 0 else 0,
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "survey-analyzer"}
