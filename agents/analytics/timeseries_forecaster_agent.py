"""
Timeseries Forecaster Agent
Analyzes time series data and forecasts trends.
"""
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from io import StringIO


def run_timeseries_forecaster(csv_content: Optional[str], table_json: Optional[List[Dict]], prompt: str) -> Dict[str, Any]:
    """
    Analyze and forecast time series data.
    Expected input: CSV with date, value, series_id columns
    or table_json with time series records.
    """
    try:
        if csv_content:
            df = pd.read_csv(StringIO(csv_content))
        elif table_json:
            df = pd.DataFrame(table_json)
        else:
            raise ValueError("timeseries-forecaster requires CSV or table_json input")

        result = {
            "total_records": len(df),
        }

        # Time series statistics
        if "date" in df.columns and "value" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                df_sorted = df.sort_values("date")
                
                result["value_statistics"] = {
                    "mean": df["value"].mean(),
                    "std": df["value"].std(),
                    "min": df["value"].min(),
                    "max": df["value"].max(),
                    "median": df["value"].median(),
                }

                # Trend analysis
                df_sorted["days_elapsed"] = (df_sorted["date"] - df_sorted["date"].min()).dt.days
                if df_sorted["days_elapsed"].max() > 0:
                    trend = (df_sorted["value"].iloc[-1] - df_sorted["value"].iloc[0]) / df_sorted["days_elapsed"].max()
                    result["trend_analysis"] = {
                        "trend_direction": "UP" if trend > 0 else "DOWN",
                        "slope": trend,
                    }

                # Volatility
                df_sorted["returns"] = df_sorted["value"].pct_change()
                result["volatility_analysis"] = {
                    "volatility": df_sorted["returns"].std(),
                    "coefficient_of_variation": df_sorted["value"].std() / (df_sorted["value"].mean() + 0.0001),
                }

                # Seasonal decomposition indicators
                result["seasonality_indicators"] = {
                    "data_points": len(df_sorted),
                    "date_range_days": (df_sorted["date"].max() - df_sorted["date"].min()).days,
                }
            except:
                pass

        # Series analysis
        if "series_id" in df.columns:
            series_dist = df["series_id"].value_counts().to_dict()
            result["by_series"] = series_dist

        # Forecast recommendations
        result["forecast_recommendations"] = {
            "suggested_models": ["ARIMA", "Exponential_Smoothing", "Prophet"] if len(df) > 20 else ["SimpleMovingAverage"],
            "data_sufficiency": "SUFFICIENT" if len(df) > 20 else "INSUFFICIENT",
            "forecasting_horizon": min(len(df) // 4, 12),
        }

        return result

    except Exception as e:
        return {"error": str(e), "agent": "timeseries-forecaster"}
