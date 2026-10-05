"""
Transcript Analyzer Agent
Analyzes meeting transcripts, call recordings, and audio transcripts for content, speakers, and topics.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_transcript_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze transcripts for speaker patterns, topic coverage, and discourse quality.
    Expected input: CSV with transcript_id, speaker, text, duration_minutes, topic columns
    or table_json with transcript records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("transcript-analyzer requires CSV or table_json input")

        result = {
            "total_transcripts": len(df),
            "unique_speakers": df.get("speaker", pd.Series()).nunique() if "speaker" in df.columns else 0,
            "total_duration_minutes": df.get("duration_minutes", pd.Series()).sum() if "duration_minutes" in df.columns else 0,
        }

        # Speaker analysis
        if "speaker" in df.columns:
            speaker_stats = df["speaker"].value_counts()
            result["speaker_count"] = len(speaker_stats)
            result["most_active_speaker"] = speaker_stats.index[0] if len(speaker_stats) > 0 else None
            result["speaker_distribution"] = speaker_stats.head(5).to_dict()

        # Text content analysis
        if "text" in df.columns:
            result["content_analysis"] = {
                "avg_text_length": df["text"].str.len().mean(),
                "total_words": df["text"].str.split().str.len().sum(),
            }

        # Topic analysis
        if "topic" in df.columns:
            topic_dist = df["topic"].value_counts().to_dict()
            result["topics_covered"] = topic_dist

        # Duration analysis
        if "duration_minutes" in df.columns:
            result["duration_analysis"] = {
                "avg_duration": df["duration_minutes"].mean(),
                "max_duration": df["duration_minutes"].max(),
                "min_duration": df["duration_minutes"].min(),
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "transcript-analyzer"}
