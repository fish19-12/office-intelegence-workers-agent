"""
Payment Optimizer Agent
Optimizes payment schedules and strategies for cost savings.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_payment_optimizer(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze and optimize payment strategies and schedules.
    Expected input: CSV with payment_id, amount, due_date, discount_available, discount_pct, terms columns
    or table_json with payment records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("payment-optimizer requires CSV or table_json input")

        result = {
            "total_payments": len(df),
            "total_amount": df.get("amount", pd.Series()).sum() if "amount" in df.columns else 0,
        }

        # Discount analysis
        if "discount_available" in df.columns and "discount_pct" in df.columns and "amount" in df.columns:
            payments_with_discount = df[df["discount_available"] == True]
            total_discount_available = (payments_with_discount["amount"] * payments_with_discount["discount_pct"] / 100).sum()
            
            result["discount_analysis"] = {
                "payments_with_discount": len(payments_with_discount),
                "total_discount_available": total_discount_available,
                "avg_discount_pct": payments_with_discount["discount_pct"].mean() if len(payments_with_discount) > 0 else 0,
            }

        # Payment terms analysis
        if "terms" in df.columns:
            terms_dist = df["terms"].value_counts().to_dict()
            result["terms_distribution"] = terms_dist

        # Payment timing optimization
        if "due_date" in df.columns:
            try:
                df["due_date"] = pd.to_datetime(df["due_date"])
                days_to_due = (df["due_date"] - pd.Timestamp.now()).dt.days
                result["payment_timing"] = {
                    "payments_due_within_7_days": (days_to_due <= 7).sum(),
                    "payments_due_within_30_days": (days_to_due <= 30).sum(),
                    "avg_days_to_due": days_to_due.mean(),
                }
            except:
                pass

        # Cash flow optimization
        result["optimization_recommendations"] = {
            "early_payment_discount_opportunities": 0,
            "payment_consolidation_candidates": 0,
            "estimated_savings": 0.0,
        }

        if "discount_available" in df.columns:
            result["optimization_recommendations"]["early_payment_discount_opportunities"] = (df["discount_available"] == True).sum()
        
        if "discount_analysis" in result:
            result["optimization_recommendations"]["estimated_savings"] = result["discount_analysis"]["total_discount_available"]

        return result

    except Exception as e:
        return {"error": str(e), "agent": "payment-optimizer"}
