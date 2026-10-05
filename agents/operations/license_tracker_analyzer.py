"""
License Tracker Agent
Tracks software licenses and compliance.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO
from datetime import datetime


def run_license_tracker_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze software licenses and usage compliance.
    Expected input: CSV with license_id, software, quantity_licensed, quantity_used, expiry_date, cost, vendor columns
    or table_json with license records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("license-tracker requires CSV or table_json input")

        result = {
            "total_licenses": len(df),
        }

        # License utilization
        if "quantity_licensed" in df.columns and "quantity_used" in df.columns:
            df["utilization"] = (df["quantity_used"] / df["quantity_licensed"] * 100).fillna(0)
            result["total_licenses_purchased"] = df["quantity_licensed"].sum()
            result["total_licenses_used"] = df["quantity_used"].sum()
            result["overall_utilization_pct"] = (result["total_licenses_used"] / (result["total_licenses_purchased"] + 0.0001) * 100)

            # Over-utilization detection
            over_utilized = df[df["utilization"] > 100]
            result["over_utilized_licenses"] = len(over_utilized)

        # Software analysis
        if "software" in df.columns:
            software_dist = df["software"].value_counts().to_dict()
            result["by_software"] = software_dist

        # Expiry analysis
        if "expiry_date" in df.columns:
            try:
                df["expiry_date"] = pd.to_datetime(df["expiry_date"])
                today = datetime.now()
                thirty_days = today.replace(day=today.day + 30 if today.day <= 1 else today.day - 1)
                
                expired = df[df["expiry_date"] < today]
                expiring_soon = df[(df["expiry_date"] >= today) & (df["expiry_date"] <= thirty_days)]
                
                result["expiry_status"] = {
                    "expired_licenses": len(expired),
                    "expiring_within_30_days": len(expiring_soon),
                    "total_valid": len(df[df["expiry_date"] >= today]),
                }
            except:
                pass

        # Cost analysis
        if "cost" in df.columns:
            result["cost_analysis"] = {
                "total_license_cost": df["cost"].sum(),
                "avg_cost_per_license": df["cost"].mean(),
                "max_cost_license": df["cost"].max(),
            }

        # Vendor analysis
        if "vendor" in df.columns:
            vendor_dist = df["vendor"].value_counts().to_dict()
            result["by_vendor"] = vendor_dist

        # Compliance assessment
        result["compliance_status"] = {
            "licenses_at_risk": len(over_utilized) if "over_utilized" in locals() else 0,
            "licenses_expiring_soon": len(expiring_soon) if "expiring_soon" in locals() else 0,
            "overall_compliance_status": "COMPLIANT" if len(df[df["utilization"] > 100]) == 0 else "NON_COMPLIANT" if "utilization" in locals() else "UNKNOWN",
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "license-tracker"}
