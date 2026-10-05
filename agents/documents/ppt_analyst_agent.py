"""
PowerPoint Presentation Analyzer Agent
Analyzes .pptx files for slide counts, text content, and presentation structure.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_ppt_analyst(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze PowerPoint presentation structure and content.
    Expected input: CSV with presentation_name, slide_count, text_content, speaker_notes columns
    or table_json with presentation metadata.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("ppt-analyst requires CSV or table_json input")

        result = {
            "total_presentations": len(df),
            "total_slides": df.get("slide_count", pd.Series()).sum() if "slide_count" in df.columns else 0,
            "avg_slides_per_presentation": df.get("slide_count", pd.Series()).mean() if "slide_count" in df.columns else 0,
        }

        # Text analysis
        if "text_content" in df.columns:
            result["content_analysis"] = {
                "avg_text_per_slide": df["text_content"].str.len().mean() / (df.get("slide_count", pd.Series()).mean() or 1),
                "presentations_with_notes": (df.get("speaker_notes", pd.Series()).notna()).sum() if "speaker_notes" in df.columns else 0,
            }

        # Presentation quality metrics
        result["presentation_quality"] = {
            "total_text_chars": df.get("text_content", pd.Series()).str.len().sum() if "text_content" in df.columns else 0,
            "avg_text_density": (df.get("text_content", pd.Series()).str.len().mean() if "text_content" in df.columns else 0),
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "ppt-analyst"}
