"""
Email Analyzer Agent
Analyzes email data for patterns, volume, sentiment, and communication trends.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_email_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze email communication patterns and content.
    Expected input: CSV with from, to, date, subject, body, sentiment columns
    or table_json with email records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("email-analyzer requires CSV or table_json input")

        result = {
            "total_emails": len(df),
            "unique_senders": df.get("from", pd.Series()).nunique() if "from" in df.columns else 0,
            "unique_recipients": df.get("to", pd.Series()).nunique() if "to" in df.columns else 0,
        }

        # Communication volume
        if "date" in df.columns:
            result["communication_volume"] = {
                "emails_per_day_avg": len(df) / max(1, df["date"].nunique()),
                "peak_communication_date": df["date"].value_counts().idxmax() if len(df) > 0 else None,
            }

        # Sentiment analysis
        if "sentiment" in df.columns:
            sentiment_dist = df["sentiment"].value_counts().to_dict()
            result["sentiment_distribution"] = sentiment_dist

        # Top senders and recipients
        if "from" in df.columns:
            top_senders = df["from"].value_counts().head(5).to_dict()
            result["top_senders"] = top_senders
        if "to" in df.columns:
            top_recipients = df["to"].value_counts().head(5).to_dict()
            result["top_recipients"] = top_recipients

        # Subject line analysis
        if "subject" in df.columns:
            result["subject_analysis"] = {
                "avg_subject_length": df["subject"].str.len().mean(),
                "subjects_with_urgent": (df["subject"].str.contains("urgent|important|asap", case=False, na=False)).sum(),
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "email-analyzer"}
