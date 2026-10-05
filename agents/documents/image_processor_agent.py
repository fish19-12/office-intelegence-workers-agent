"""
Image Processor Agent
Analyzes images and metadata for image-based data intelligence.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_image_processor(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Process and analyze image metadata.
    Expected input: CSV with image_id, filename, file_size, width, height, format, color_space columns
    or table_json with image metadata.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("image-processor requires CSV or table_json input")

        result = {
            "total_images": len(df),
        }

        # File format analysis
        if "format" in df.columns:
            format_dist = df["format"].value_counts().to_dict()
            result["by_format"] = format_dist

        # File size analysis
        if "file_size" in df.columns:
            result["file_size_analysis"] = {
                "total_size_mb": df["file_size"].sum() / (1024 * 1024),
                "avg_size_kb": df["file_size"].mean() / 1024,
                "max_size_mb": df["file_size"].max() / (1024 * 1024),
                "min_size_kb": df["file_size"].min() / 1024,
            }

        # Image dimension analysis
        if "width" in df.columns and "height" in df.columns:
            df["aspect_ratio"] = df["width"] / (df["height"] + 0.0001)
            result["image_dimensions"] = {
                "avg_width": df["width"].mean(),
                "avg_height": df["height"].mean(),
                "most_common_aspect_ratio": df["aspect_ratio"].mode()[0] if len(df["aspect_ratio"].mode()) > 0 else None,
            }

        # Color space analysis
        if "color_space" in df.columns:
            color_dist = df["color_space"].value_counts().to_dict()
            result["by_color_space"] = color_dist

        # Resolution quality classification
        if "width" in df.columns and "height" in df.columns:
            df["resolution_type"] = pd.cut(
                df["width"] * df["height"],
                bins=[0, 480*360, 1280*720, 1920*1080, float('inf')],
                labels=["Low", "SD", "HD", "UHD"]
            )
            resolution_dist = df["resolution_type"].value_counts().to_dict()
            result["by_resolution"] = {str(k): v for k, v in resolution_dist.items()}

        return result

    except Exception as e:
        return {"error": str(e), "agent": "image-processor"}
