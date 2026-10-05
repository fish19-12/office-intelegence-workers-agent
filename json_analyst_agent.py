from __future__ import annotations

import json as json_module
from typing import Any, Dict, List, Optional

import pandas as pd


JSON_ANALYST_PROMPT = """You are an elite JSON data analyst focused on structure, quality, and actionable insight.

MISSION
- Parse and validate the JSON payload before drawing conclusions.
- Flatten nested structures when helpful and keep track of source relationships.
- Identify missing values, schema drift, malformed records, and anomalies.
- Summarize the dataset in a business-friendly way with concrete evidence.

OPERATING PROTOCOL
1. Inspect whether the payload is a single object, list of objects, or nested structure.
2. Check for structural inconsistencies and data quality issues.
3. Flatten nested keys for analysis while preserving interpretability.
4. Compute sample statistics and highlight notable patterns.
5. Recommend normalization, validation, or schema changes when appropriate.

OUTPUT STANDARDS
- Give a concise summary of dataset shape and quality.
- Highlight the most important columns, nulls, and anomalies.
- Recommend next steps for normalization or schema cleanup.
- If the JSON is invalid or unsupported, explain exactly why and what format is required.

QUALITY GATES
- Do not infer missing fields as valid values.
- Distinguish between observed data and inferred business assumptions.
- Prefer explicit evidence over generic relational assumptions.
"""


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Recursively flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def analyze_json_data(json_text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Analyze JSON data - flatten, compute stats, detect issues."""
    try:
        data = json_module.loads(json_text)
        
        # Handle both list of objects and single object
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            return {"error": "JSON must be an object or array of objects."}
        
        # Flatten records
        flattened = [flatten_dict(r) for r in records]
        df = pd.DataFrame(flattened)
        
        return {
            "record_count": len(records),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
            "null_counts": df.isna().sum().to_dict(),
            "sample": df.head(3).to_dict(orient="records"),
            "recommendation": f"Flattened {len(records)} records into {len(df.columns)} columns. Check null_counts for data quality.",
        }
    except json_module.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


def run_json_analyst(json_text: Optional[str], prompt: str) -> Dict[str, Any]:
    if json_text:
        return analyze_json_data(json_text, prompt)
    return {"error": "json-analyst requires JSON input."}
