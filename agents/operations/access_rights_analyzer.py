"""
Access Rights Analyzer Agent
Analyzes user access rights and security compliance.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_access_rights_analyzer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze user access rights and permissions.
    Expected input: CSV with user_id, username, role, department, access_level, last_login, status columns
    or table_json with access records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("access-rights requires CSV or table_json input")

        result = {
            "total_users": len(df),
        }

        # Role distribution
        if "role" in df.columns:
            role_dist = df["role"].value_counts().to_dict()
            result["role_distribution"] = role_dist

        # Access level analysis
        if "access_level" in df.columns:
            access_dist = df["access_level"].value_counts().to_dict()
            result["access_level_distribution"] = access_dist

        # Department analysis
        if "department" in df.columns:
            dept_dist = df["department"].value_counts().to_dict()
            result["by_department"] = dept_dist

        # User status
        if "status" in df.columns:
            status_dist = df["status"].value_counts().to_dict()
            result["status_distribution"] = status_dist
            result["active_users"] = (df["status"] == "active").sum() if "status" in df.columns else 0
            result["inactive_users"] = (df["status"] == "inactive").sum() if "status" in df.columns else 0

        # Privileged user analysis
        if "access_level" in df.columns:
            privileged_users = df[df["access_level"].isin(["admin", "super_admin", "privileged"])]
            result["privileged_access"] = {
                "privileged_users": len(privileged_users),
                "privileged_user_percentage": (len(privileged_users) / len(df) * 100) if len(df) > 0 else 0,
            }

        # Login activity
        if "last_login" in df.columns:
            try:
                df["last_login"] = pd.to_datetime(df["last_login"])
                from datetime import datetime, timedelta
                today = datetime.now()
                inactive_30 = (today - df["last_login"]).dt.days > 30
                result["login_activity"] = {
                    "users_inactive_30_days": inactive_30.sum(),
                    "users_inactive_90_days": ((today - df["last_login"]).dt.days > 90).sum(),
                }
            except:
                pass

        # Security risk assessment
        if "access_level" in df.columns and "status" in df.columns:
            high_privilege_inactive = df[(df["access_level"].isin(["admin", "super_admin"])) & (df["status"] == "inactive")]
            result["security_risks"] = {
                "high_privilege_inactive": len(high_privilege_inactive),
            }

        # Role-based access control
        if "role" in df.columns and "access_level" in df.columns:
            role_access = df.groupby("role")["access_level"].value_counts().to_dict()
            result["rbac_matrix"] = role_access

        return result

    except Exception as e:
        return {"error": str(e), "agent": "access-rights"}
