"""
PDF Extractor Agent
Extracts and analyzes data from PDF documents.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_pdf_extractor(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Extract and analyze PDF document data.
    Expected input: CSV with pdf_id, filename, page_count, text_content, tables_found, file_size columns
    or table_json with PDF metadata.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("pdf-extractor requires CSV or table_json input")

        result = {
            "total_pdfs": len(df),
        }

        # Page analysis
        if "page_count" in df.columns:
            result["page_analysis"] = {
                "total_pages": df["page_count"].sum(),
                "avg_pages_per_pdf": df["page_count"].mean(),
                "max_pages": df["page_count"].max(),
                "min_pages": df["page_count"].min(),
            }

        # Text content analysis
        if "text_content" in df.columns:
            result["text_analysis"] = {
                "total_text_chars": df["text_content"].str.len().sum(),
                "avg_chars_per_pdf": df["text_content"].str.len().mean(),
                "empty_pdfs": (df["text_content"].str.len() == 0).sum(),
            }

        # Table detection
        if "tables_found" in df.columns:
            result["table_extraction"] = {
                "total_tables": df["tables_found"].sum(),
                "pdfs_with_tables": (df["tables_found"] > 0).sum(),
                "avg_tables_per_pdf": df["tables_found"].mean(),
            }

        # File size analysis
        if "file_size" in df.columns:
            result["file_size_analysis"] = {
                "total_size_mb": df["file_size"].sum() / (1024 * 1024),
                "avg_size_kb": df["file_size"].mean() / 1024,
            }

        # Filename analysis
        if "filename" in df.columns:
            result["filename_analysis"] = {
                "unique_filenames": df["filename"].nunique(),
                "duplicate_filenames": len(df) - df["filename"].nunique(),
            }

        # PDF quality assessment
        if "page_count" in df.columns and "text_content" in df.columns:
            result["pdf_quality"] = {
                "high_quality": ((df["text_content"].str.len() / (df["page_count"] + 0.0001)) > 100).sum(),
                "medium_quality": ((df["text_content"].str.len() / (df["page_count"] + 0.0001)).between(50, 100)).sum(),
                "low_quality": ((df["text_content"].str.len() / (df["page_count"] + 0.0001)) < 50).sum(),
            }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "pdf-extractor"}
