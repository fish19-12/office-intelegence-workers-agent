 """
Campaign Performance Analyzer Agent

Advanced marketing campaign intelligence agent.

Analyzes:
- Campaign performance
- Marketing spend
- Impressions
- Clicks
- CTR
- CPC
- CPM
- Conversions
- Conversion rate
- CPA
- Revenue
- ROAS
- ROI
- Profit
- Profit margin
- Channel performance
- Campaign rankings
- Underperforming campaigns
- Budget efficiency
- Anomalies
- Data quality
- Trends
- Actionable recommendations

Expected input:
CSV or table_json containing campaign records.

Common columns:
campaign_id
campaign_name
channel
spent
impressions
clicks
conversions
revenue
date / campaign_date / start_date / end_date
"""

import json
import math
from typing import Optional, List, Dict, Any
from io import StringIO

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

AGENT_NAME = "campaign-performance"

DEFAULT_TOP_N = 5

EPSILON = 0.0000001


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(value: Any) -> float:
    """
    Convert a value to a JSON-safe float.
    """
    try:
        if value is None:
            return 0.0

        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return 0.0

        return number

    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    """
    Convert a value to a safe integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_divide(
    numerator: float,
    denominator: float,
    multiplier: float = 1.0,
) -> float:
    """
    Safely divide two numbers.

    Returns 0 when denominator is zero.
    """
    try:
        denominator = float(denominator)
        numerator = float(numerator)

        if denominator == 0:
            return 0.0

        result = (numerator / denominator) * multiplier

        if math.isnan(result) or math.isinf(result):
            return 0.0

        return float(result)

    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _clean_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Convert known numeric columns to numeric values.

    Invalid values become NaN and are later handled safely.
    """
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def _round_dataframe_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """
    cleaned = []

    for record in records:
        clean_record = {}

        for key, value in record.items():

            if pd.isna(value):
                clean_record[key] = None

            elif isinstance(value, (int, float)):
                clean_record[key] = _safe_float(value)

            else:
                clean_record[key] = value

        cleaned.append(clean_record)

    return cleaned


def _json_safe(value: Any) -> Any:
    """
    Recursively convert pandas/numpy values into JSON-safe values.
    """
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            _json_safe(v)
            for v in value
        ]

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (int, float)):
        return _safe_float(value)

    return value


# ============================================================
# COLUMN DISCOVERY
# ============================================================

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find a likely campaign date column.
    """
    candidates = [
        "date",
        "campaign_date",
        "start_date",
        "end_date",
        "created_at",
        "created_date",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def _find_campaign_identifier(df: pd.DataFrame) -> Optional[str]:
    """
    Find the best campaign identifier.
    """
    candidates = [
        "campaign_id",
        "campaign_name",
        "name",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# DATA QUALITY
# ============================================================

def _analyze_data_quality(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    expected_columns = [
        "campaign_id",
        "campaign_name",
        "channel",
        "spent",
        "impressions",
        "clicks",
        "conversions",
        "revenue",
    ]

    available_metrics = [
        column
        for column in expected_columns
        if column in df.columns
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    missing_values = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            missing_values[column] = {
                "count": missing_count,
                "percentage": _safe_divide(
                    missing_count,
                    len(df),
                    100,
                ),
            }

    duplicate_rows = int(df.duplicated().sum())

    negative_values = {}

    numeric_columns = [
        "spent",
        "impressions",
        "clicks",
        "conversions",
        "revenue",
    ]

    for column in numeric_columns:
        if column in df.columns:

            count = int(
                (df[column].fillna(0) < 0).sum()
            )

            if count > 0:
                negative_values[column] = count

    warnings = []

    if missing_columns:
        warnings.append(
            "Some recommended campaign columns are missing. "
            "Certain metrics cannot be calculated."
        )

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows} duplicate row(s) detected."
        )

    if negative_values:
        warnings.append(
            "Negative marketing metrics were detected."
        )

    return {
        "available_columns": available_metrics,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "negative_values": negative_values,
        "warnings": warnings,
    }


# ============================================================
# CAMPAIGN PERFORMANCE METRICS
# ============================================================

def _calculate_campaign_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    if "spent" in result.columns:
        result["spent"] = result["spent"].fillna(0)

    if "impressions" in result.columns:
        result["impressions"] = result["impressions"].fillna(0)

    if "clicks" in result.columns:
        result["clicks"] = result["clicks"].fillna(0)

    if "conversions" in result.columns:
        result["conversions"] = result["conversions"].fillna(0)

    if "revenue" in result.columns:
        result["revenue"] = result["revenue"].fillna(0)

    if "spent" in result.columns and "revenue" in result.columns:

        result["profit"] = (
            result["revenue"] -
            result["spent"]
        )

        result["roi_percentage"] = result.apply(
            lambda row: _safe_divide(
                row["profit"],
                row["spent"],
                100,
            ),
            axis=1,
        )

        result["roas"] = result.apply(
            lambda row: _safe_divide(
                row["revenue"],
                row["spent"],
            ),
            axis=1,
        )

        result["profit_margin_pct"] = result.apply(
            lambda row: _safe_divide(
                row["profit"],
                row["revenue"],
                100,
            ),
            axis=1,
        )

    if "clicks" in result.columns and "impressions" in result.columns:

        result["ctr_pct"] = result.apply(
            lambda row: _safe_divide(
                row["clicks"],
                row["impressions"],
                100,
            ),
            axis=1,
        )

    if "spent" in result.columns and "clicks" in result.columns:

        result["cpc"] = result.apply(
            lambda row: _safe_divide(
                row["spent"],
                row["clicks"],
            ),
            axis=1,
        )

    if "spent" in result.columns and "impressions" in result.columns:

        result["cpm"] = result.apply(
            lambda row: _safe_divide(
                row["spent"],
                row["impressions"],
                1000,
            ),
            axis=1,
        )

    if "conversions" in result.columns and "clicks" in result.columns:

        result["conversion_rate_pct"] = result.apply(
            lambda row: _safe_divide(
                row["conversions"],
                row["clicks"],
                100,
            ),
            axis=1,
        )

    if "spent" in result.columns and "conversions" in result.columns:

        result["cpa"] = result.apply(
            lambda row: _safe_divide(
                row["spent"],
                row["conversions"],
            ),
            axis=1,
        )

    if "revenue" in result.columns and "clicks" in result.columns:

        result["revenue_per_click"] = result.apply(
            lambda row: _safe_divide(
                row["revenue"],
                row["clicks"],
            ),
            axis=1,
        )

    if "revenue" in result.columns and "conversions" in result.columns:

        result["revenue_per_conversion"] = result.apply(
            lambda row: _safe_divide(
                row["revenue"],
                row["conversions"],
            ),
            axis=1,
        )

    return result


# ============================================================
# MAIN AGENT
# ============================================================

def run_campaign_performance_analyzer(
    csv_content: Optional[str],
    table_json: Optional[List[Dict]],
    prompt: str,
) -> Dict[str, Any]:
    """
    Analyze marketing campaign performance and ROI.

    Supports:

    CSV:
        campaign_id
        campaign_name
        channel
        spent
        impressions
        clicks
        conversions
        revenue

    Optional date columns:
        date
        campaign_date
        start_date
        end_date

    Or table_json with campaign records.
    """

    try:

        # ====================================================
        # INPUT
        # ====================================================

        if csv_content:

            df = pd.read_csv(
                StringIO(csv_content)
            )

        elif table_json:

            df = pd.DataFrame(
                table_json
            )

        else:

            raise ValueError(
                "campaign-performance requires CSV "
                "or table_json input"
            )

        if df.empty:

            return {
                "agent": AGENT_NAME,
                "error": "The campaign dataset is empty.",
                "total_campaigns": 0,
            }

        # Normalize column names.
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        # ====================================================
        # DATA QUALITY
        # ====================================================

        data_quality = _analyze_data_quality(
            df
        )

        # ====================================================
        # NUMERIC CLEANING
        # ====================================================

        numeric_columns = [
            "spent",
            "impressions",
            "clicks",
            "conversions",
            "revenue",
        ]

        df = _clean_numeric_columns(
            df,
            numeric_columns,
        )

        # ====================================================
        # CAMPAIGN METRICS
        # ====================================================

        metrics_df = _calculate_campaign_metrics(
            df
        )

        # ====================================================
        # BASE RESULT
        # ====================================================

        result: Dict[str, Any] = {

            "agent": AGENT_NAME,

            "total_campaigns": int(
                len(metrics_df)
            ),

            "columns": [
                str(column)
                for column in metrics_df.columns
            ],

            "data_quality": data_quality,

        }

        # ====================================================
        # BUDGET ANALYSIS
        # ====================================================

        if "spent" in metrics_df.columns:

            total_spend = _safe_float(
                metrics_df["spent"]
                .fillna(0)
                .sum()
            )

            result["total_spend"] = total_spend

            result["avg_spend_per_campaign"] = (
                _safe_divide(
                    total_spend,
                    len(metrics_df),
                )
            )

            result["median_spend_per_campaign"] = (
                _safe_float(
                    metrics_df["spent"]
                    .fillna(0)
                    .median()
                )
            )

            result["largest_campaign_spend"] = (
                _safe_float(
                    metrics_df["spent"]
                    .fillna(0)
                    .max()
                )
            )

        # ====================================================
        # ENGAGEMENT METRICS
        # ====================================================

        if "impressions" in metrics_df.columns:

            total_impressions = _safe_float(
                metrics_df["impressions"]
                .fillna(0)
                .sum()
            )

            result["total_impressions"] = (
                total_impressions
            )

        if "clicks" in metrics_df.columns:

            total_clicks = _safe_float(
                metrics_df["clicks"]
                .fillna(0)
                .sum()
            )

            result["total_clicks"] = total_clicks

        # ====================================================
        # CTR
        # ====================================================

        if (
            "clicks" in metrics_df.columns
            and
            "impressions" in metrics_df.columns
        ):

            result["click_through_rate_pct"] = (
                _safe_divide(
                    metrics_df["clicks"].sum(),
                    metrics_df["impressions"].sum(),
                    100,
                )
            )

        # ====================================================
        # ADVANCED ADVERTISING METRICS
        # ====================================================

        if (
            "spent" in metrics_df.columns
            and
            "clicks" in metrics_df.columns
        ):

            result["cpc"] = _safe_divide(
                metrics_df["spent"].sum(),
                metrics_df["clicks"].sum(),
            )

        if (
            "spent" in metrics_df.columns
            and
            "impressions" in metrics_df.columns
        ):

            result["cpm"] = _safe_divide(
                metrics_df["spent"].sum(),
                metrics_df["impressions"].sum(),
                1000,
            )

        # ====================================================
        # CONVERSION ANALYSIS
        # ====================================================

        if "conversions" in metrics_df.columns:

            total_conversions = _safe_float(
                metrics_df["conversions"]
                .fillna(0)
                .sum()
            )

            result["total_conversions"] = (
                total_conversions
            )

            if "clicks" in metrics_df.columns:

                result["conversion_rate_pct"] = (
                    _safe_divide(
                        total_conversions,
                        metrics_df["clicks"].sum(),
                        100,
                    )
                )

        # ====================================================
        # REVENUE ANALYSIS
        # ====================================================

        if "revenue" in metrics_df.columns:

            total_revenue = _safe_float(
                metrics_df["revenue"]
                .fillna(0)
                .sum()
            )

            result["total_revenue"] = (
                total_revenue
            )

            if "clicks" in metrics_df.columns:

                result["revenue_per_click"] = (
                    _safe_divide(
                        total_revenue,
                        metrics_df["clicks"].sum(),
                    )
                )

            if "conversions" in metrics_df.columns:

                result["revenue_per_conversion"] = (
                    _safe_divide(
                        total_revenue,
                        metrics_df["conversions"].sum(),
                    )
                )

        # ====================================================
        # ROI / ROAS
        # ====================================================

        if (
            "spent" in metrics_df.columns
            and
            "revenue" in metrics_df.columns
        ):

            total_revenue = _safe_float(
                metrics_df["revenue"]
                .fillna(0)
                .sum()
            )

            total_spent = _safe_float(
                metrics_df["spent"]
                .fillna(0)
                .sum()
            )

            net_profit = (
                total_revenue -
                total_spent
            )

            roi_percentage = _safe_divide(
                net_profit,
                total_spent,
                100,
            )

            roas = _safe_divide(
                total_revenue,
                total_spent,
            )

            profit_margin = _safe_divide(
                net_profit,
                total_revenue,
                100,
            )

            result["roi_analysis"] = {

                "total_revenue": total_revenue,

                "total_investment": total_spent,

                "net_profit": net_profit,

                "roi_percentage": roi_percentage,

                "roas": roas,

                "profit_margin_pct": profit_margin,

            }

        # ====================================================
        # CPA ANALYSIS
        # ====================================================

        if (
            "spent" in metrics_df.columns
            and
            "conversions" in metrics_df.columns
        ):

            total_spend = _safe_float(
                metrics_df["spent"]
                .fillna(0)
                .sum()
            )

            total_conversions = _safe_float(
                metrics_df["conversions"]
                .fillna(0)
                .sum()
            )

            result["cpa_analysis"] = {

                "avg_cost_per_acquisition": (
                    _safe_divide(
                        total_spend,
                        total_conversions,
                    )
                ),

                "total_spend": total_spend,

                "total_conversions": total_conversions,

            }

        # ====================================================
        # CHANNEL ANALYSIS
        # ====================================================

        if "channel" in metrics_df.columns:

            channel_groups = []

            for channel, group in metrics_df.groupby(
                "channel",
                dropna=False,
            ):

                channel_name = (
                    "Unknown"
                    if pd.isna(channel)
                    else str(channel)
                )

                spend = (
                    group["spent"].sum()
                    if "spent" in group.columns
                    else 0
                )

                impressions = (
                    group["impressions"].sum()
                    if "impressions" in group.columns
                    else 0
                )

                clicks = (
                    group["clicks"].sum()
                    if "clicks" in group.columns
                    else 0
                )

                conversions = (
                    group["conversions"].sum()
                    if "conversions" in group.columns
                    else 0
                )

                revenue = (
                    group["revenue"].sum()
                    if "revenue" in group.columns
                    else 0
                )

                profit = (
                    revenue -
                    spend
                )

                channel_result = {

                    "channel": channel_name,

                    "campaigns": int(len(group)),

                    "spend": _safe_float(spend),

                    "impressions": _safe_float(
                        impressions
                    ),

                    "clicks": _safe_float(
                        clicks
                    ),

                    "conversions": _safe_float(
                        conversions
                    ),

                    "revenue": _safe_float(
                        revenue
                    ),

                    "profit": _safe_float(
                        profit
                    ),

                    "ctr_pct": _safe_divide(
                        clicks,
                        impressions,
                        100,
                    ),

                    "conversion_rate_pct": _safe_divide(
                        conversions,
                        clicks,
                        100,
                    ),

                    "cpc": _safe_divide(
                        spend,
                        clicks,
                    ),

                    "cpm": _safe_divide(
                        spend,
                        impressions,
                        1000,
                    ),

                    "cpa": _safe_divide(
                        spend,
                        conversions,
                    ),

                    "roas": _safe_divide(
                        revenue,
                        spend,
                    ),

                    "roi_percentage": _safe_divide(
                        profit,
                        spend,
                        100,
                    ),

                }

                channel_groups.append(
                    channel_result
                )

            # Preserve your original by_channel result.
            result["by_channel"] = {
                item["channel"]: {
                    key: value
                    for key, value in item.items()
                    if key != "channel"
                }
                for item in channel_groups
            }

            # Advanced channel ranking.
            result["channel_performance"] = sorted(
                channel_groups,
                key=lambda item: item.get(
                    "roas",
                    0,
                ),
                reverse=True,
            )

            if channel_groups:

                result["best_channel_by_roas"] = (
                    max(
                        channel_groups,
                        key=lambda item: item.get(
                            "roas",
                            0,
                        ),
                    )
                )

                result["best_channel_by_conversion_rate"] = (
                    max(
                        channel_groups,
                        key=lambda item: item.get(
                            "conversion_rate_pct",
                            0,
                        ),
                    )
                )

                result["best_channel_by_profit"] = (
                    max(
                        channel_groups,
                        key=lambda item: item.get(
                            "profit",
                            0,
                        ),
                    )
                )

        # ====================================================
        # CAMPAIGN PERFORMANCE RANKING
        # ====================================================

        campaign_id_column = (
            _find_campaign_identifier(
                metrics_df
            )
        )

        if (
            "spent" in metrics_df.columns
            and
            "revenue" in metrics_df.columns
        ):

            ranking_columns = []

            if campaign_id_column:
                ranking_columns.append(
                    campaign_id_column
                )

            for column in [
                "channel",
                "spent",
                "revenue",
                "profit",
                "roas",
                "roi_percentage",
                "cpa",
                "ctr_pct",
                "conversion_rate_pct",
            ]:

                if column in metrics_df.columns:
                    ranking_columns.append(
                        column
                    )

            ranking_df = metrics_df.copy()

            top_performers = (
                ranking_df
                .sort_values(
                    "roas",
                    ascending=False,
                )
                .head(DEFAULT_TOP_N)
            )

            bottom_performers = (
                ranking_df
                .sort_values(
                    "roas",
                    ascending=True,
                )
                .head(DEFAULT_TOP_N)
            )

            result["top_performing_campaigns"] = (
                _round_dataframe_records(
                    top_performers[
                        ranking_columns
                    ].to_dict("records")
                )
            )

            result["lowest_performing_campaigns"] = (
                _round_dataframe_records(
                    bottom_performers[
                        ranking_columns
                    ].to_dict("records")
                )
            )

        # ====================================================
        # UNDERPERFORMING CAMPAIGNS
        # ====================================================

        underperforming = []

        if (
            "spent" in metrics_df.columns
            and
            "revenue" in metrics_df.columns
        ):

            for index, row in metrics_df.iterrows():

                spend = _safe_float(
                    row.get("spent", 0)
                )

                revenue = _safe_float(
                    row.get("revenue", 0)
                )

                conversions = _safe_float(
                    row.get("conversions", 0)
                )

                roas = _safe_float(
                    row.get("roas", 0)
                )

                reasons = []

                if spend > 0 and revenue < spend:
                    reasons.append(
                        "Revenue is below advertising spend"
                    )

                if (
                    spend > 0
                    and conversions == 0
                ):
                    reasons.append(
                        "Campaign has spending but no conversions"
                    )

                if roas < 1 and spend > 0:
                    reasons.append(
                        "ROAS is below break-even"
                    )

                if reasons:

                    campaign_info = {
                        "row_index": int(index),
                        "reasons": reasons,
                        "spent": spend,
                        "revenue": revenue,
                        "conversions": conversions,
                        "roas": roas,
                    }

                    if campaign_id_column:
                        campaign_info[
                            campaign_id_column
                        ] = row.get(
                            campaign_id_column
                        )

                    if "channel" in metrics_df.columns:
                        campaign_info["channel"] = (
                            row.get("channel")
                        )

                    underperforming.append(
                        campaign_info
                    )

        result["underperforming_campaigns"] = (
            underperforming
        )

        # ====================================================
        # ZERO-CONVERSION CAMPAIGNS
        # ====================================================

        zero_conversion_campaigns = []

        if (
            "conversions" in metrics_df.columns
            and
            "spent" in metrics_df.columns
        ):

            zero_df = metrics_df[
                (
                    metrics_df["conversions"]
                    .fillna(0)
                    <= 0
                )
                &
                (
                    metrics_df["spent"]
                    .fillna(0)
                    > 0
                )
            ]

            for index, row in zero_df.iterrows():

                item = {
                    "row_index": int(index),
                    "spent": _safe_float(
                        row.get("spent", 0)
                    ),
                }

                if campaign_id_column:
                    item[
                        campaign_id_column
                    ] = row.get(
                        campaign_id_column
                    )

                if "channel" in metrics_df.columns:
                    item["channel"] = row.get(
                        "channel"
                    )

                zero_conversion_campaigns.append(
                    item
                )

        result["zero_conversion_campaigns"] = (
            zero_conversion_campaigns
        )

        # ====================================================
        # BUDGET CONCENTRATION
        # ====================================================

        if "spent" in metrics_df.columns:

            total_spend = _safe_float(
                metrics_df["spent"]
                .fillna(0)
                .sum()
            )

            if total_spend > 0:

                concentration_df = (
                    metrics_df
                    .sort_values(
                        "spent",
                        ascending=False,
                    )
                    .copy()
                )

                concentration_df[
                    "spend_percentage"
                ] = (
                    concentration_df["spent"]
                    / total_spend
                    * 100
                )

                top_5_spend = _safe_float(
                    concentration_df
                    .head(5)["spent"]
                    .sum()
                )

                result["budget_concentration"] = {

                    "top_5_campaign_spend": (
                        top_5_spend
                    ),

                    "top_5_spend_percentage": (
                        _safe_divide(
                            top_5_spend,
                            total_spend,
                            100,
                        )
                    ),

                    "largest_campaign_spend": (
                        _safe_float(
                            concentration_df
                            .iloc[0]["spent"]
                        )
                    ),

                    "largest_campaign_spend_percentage": (
                        _safe_divide(
                            concentration_df
                            .iloc[0]["spent"],
                            total_spend,
                            100,
                        )
                    ),

                }

        # ====================================================
        # EFFICIENCY SCORE
        # ====================================================

        if len(metrics_df) > 0:

            efficiency_df = metrics_df.copy()

            score_components = []

            if "roas" in efficiency_df.columns:

                efficiency_df[
                    "roas_rank"
                ] = efficiency_df[
                    "roas"
                ].rank(
                    pct=True
                )

                score_components.append(
                    efficiency_df[
                        "roas_rank"
                    ] * 40
                )

            if "conversion_rate_pct" in efficiency_df.columns:

                efficiency_df[
                    "conversion_rank"
                ] = efficiency_df[
                    "conversion_rate_pct"
                ].rank(
                    pct=True
                )

                score_components.append(
                    efficiency_df[
                        "conversion_rank"
                    ] * 30
                )

            if "ctr_pct" in efficiency_df.columns:

                efficiency_df[
                    "ctr_rank"
                ] = efficiency_df[
                    "ctr_pct"
                ].rank(
                    pct=True
                )

                score_components.append(
                    efficiency_df[
                        "ctr_rank"
                    ] * 15
                )

            if "cpa" in efficiency_df.columns:

                efficiency_df[
                    "cpa_rank"
                ] = efficiency_df[
                    "cpa"
                ].rank(
                    pct=True,
                    ascending=False,
                )

                score_components.append(
                    efficiency_df[
                        "cpa_rank"
                    ] * 15
                )

            if score_components:

                efficiency_df[
                    "efficiency_score"
                ] = sum(
                    score_components
                )

                efficiency_df[
                    "efficiency_score"
                ] = (
                    efficiency_df[
                        "efficiency_score"
                    ]
                    .fillna(0)
                    .clip(
                        lower=0,
                        upper=100,
                    )
                )

                efficiency_columns = []

                if campaign_id_column:
                    efficiency_columns.append(
                        campaign_id_column
                    )

                if "channel" in efficiency_df.columns:
                    efficiency_columns.append(
                        "channel"
                    )

                efficiency_columns.append(
                    "efficiency_score"
                )

                for metric in [
                    "roas",
                    "roi_percentage",
                    "cpa",
                    "ctr_pct",
                    "conversion_rate_pct",
                ]:
                    if metric in efficiency_df.columns:
                        efficiency_columns.append(
                            metric
                        )

                result["campaign_efficiency_ranking"] = (
                    _round_dataframe_records(
                        efficiency_df
                        .sort_values(
                            "efficiency_score",
                            ascending=False,
                        )
                        .head(10)[
                            efficiency_columns
                        ]
                        .to_dict("records")
                    )
                )

        # ====================================================
        # ANOMALY DETECTION
        # ====================================================

        anomalies = []

        # Spend anomalies
        if "spent" in metrics_df.columns:

            spend_series = metrics_df[
                "spent"
            ].fillna(0)

            if len(spend_series) >= 3:

                mean_spend = spend_series.mean()
                std_spend = spend_series.std()

                if std_spend > 0:

                    for index, row in metrics_df.iterrows():

                        spend = _safe_float(
                            row.get("spent", 0)
                        )

                        if spend > (
                            mean_spend +
                            2 * std_spend
                        ):

                            anomaly = {
                                "type": "high_spend",
                                "row_index": int(index),
                                "value": spend,
                                "threshold": _safe_float(
                                    mean_spend +
                                    2 * std_spend
                                ),
                            }

                            if campaign_id_column:
                                anomaly[
                                    campaign_id_column
                                ] = row.get(
                                    campaign_id_column
                                )

                            anomalies.append(
                                anomaly
                            )

        # Conversion anomalies
        if "conversion_rate_pct" in metrics_df.columns:

            conversion_rates = (
                metrics_df[
                    "conversion_rate_pct"
                ]
                .replace(
                    [float("inf"), -float("inf")],
                    0,
                )
                .fillna(0)
            )

            if len(conversion_rates) >= 3:

                mean_conversion = (
                    conversion_rates.mean()
                )

                std_conversion = (
                    conversion_rates.std()
                )

                if std_conversion > 0:

                    for index, row in metrics_df.iterrows():

                        conversion_rate = (
                            _safe_float(
                                row.get(
                                    "conversion_rate_pct",
                                    0,
                                )
                            )
                        )

                        if conversion_rate > (
                            mean_conversion +
                            2 * std_conversion
                        ):

                            anomaly = {
                                "type": (
                                    "unusually_high_conversion_rate"
                                ),
                                "row_index": int(index),
                                "value": conversion_rate,
                            }

                            if campaign_id_column:
                                anomaly[
                                    campaign_id_column
                                ] = row.get(
                                    campaign_id_column
                                )

                            anomalies.append(
                                anomaly
                            )

        result["anomalies"] = anomalies

        # ====================================================
        # DATE / TREND ANALYSIS
        # ====================================================

        date_column = _find_date_column(
            metrics_df
        )

        if date_column:

            trend_df = metrics_df.copy()

            trend_df[
                date_column
            ] = pd.to_datetime(
                trend_df[
                    date_column
                ],
                errors="coerce",
            )

            valid_dates = trend_df[
                trend_df[
                    date_column
                ].notna()
            ].copy()

            if not valid_dates.empty:

                valid_dates[
                    "period"
                ] = valid_dates[
                    date_column
                ].dt.to_period(
                    "M"
                ).astype(str)

                aggregation = {}

                for metric in [
                    "spent",
                    "impressions",
                    "clicks",
                    "conversions",
                    "revenue",
                ]:

                    if metric in valid_dates.columns:
                        aggregation[
                            metric
                        ] = "sum"

                if aggregation:

                    trend_summary = (
                        valid_dates
                        .groupby("period")
                        .agg(aggregation)
                        .reset_index()
                    )

                    if (
                        "clicks" in trend_summary.columns
                        and
                        "impressions" in trend_summary.columns
                    ):

                        trend_summary[
                            "ctr_pct"
                        ] = trend_summary.apply(
                            lambda row: _safe_divide(
                                row["clicks"],
                                row["impressions"],
                                100,
                            ),
                            axis=1,
                        )

                    if (
                        "conversions" in trend_summary.columns
                        and
                        "clicks" in trend_summary.columns
                    ):

                        trend_summary[
                            "conversion_rate_pct"
                        ] = trend_summary.apply(
                            lambda row: _safe_divide(
                                row["conversions"],
                                row["clicks"],
                                100,
                            ),
                            axis=1,
                        )

                    if (
                        "revenue" in trend_summary.columns
                        and
                        "spent" in trend_summary.columns
                    ):

                        trend_summary[
                            "roas"
                        ] = trend_summary.apply(
                            lambda row: _safe_divide(
                                row["revenue"],
                                row["spent"],
                            ),
                            axis=1,
                        )

                    result["monthly_trend"] = (
                        _round_dataframe_records(
                            trend_summary.to_dict(
                                "records"
                            )
                        )
                    )

        # ====================================================
        # EXECUTIVE SUMMARY
        # ====================================================

        summary = []

        if (
            "roi_analysis" in result
        ):

            roi = result[
                "roi_analysis"
            ]

            summary.append(
                f"Total campaign revenue is "
                f"{roi['total_revenue']:.2f} "
                f"against "
                f"{roi['total_investment']:.2f} "
                f"in advertising spend."
            )

            summary.append(
                f"Overall ROAS is "
                f"{roi['roas']:.2f}x "
                f"with ROI of "
                f"{roi['roi_percentage']:.2f}%."
            )

        if (
            "underperforming_campaigns"
            in result
            and
            result[
                "underperforming_campaigns"
            ]
        ):

            summary.append(
                f"{len(result['underperforming_campaigns'])} "
                "campaign(s) require performance review."
            )

        if (
            "zero_conversion_campaigns"
            in result
            and
            result[
                "zero_conversion_campaigns"
            ]
        ):

            summary.append(
                f"{len(result['zero_conversion_campaigns'])} "
                "campaign(s) have spend but no conversions."
            )

        if (
            "best_channel_by_roas"
            in result
        ):

            best_channel = result[
                "best_channel_by_roas"
            ]

            summary.append(
                f"{best_channel['channel']} "
                "currently has the highest channel ROAS."
            )

        if (
            "budget_concentration"
            in result
        ):

            concentration = result[
                "budget_concentration"
            ][
                "top_5_spend_percentage"
            ]

            if concentration >= 60:

                summary.append(
                    f"The top 5 campaigns consume "
                    f"{concentration:.2f}% "
                    "of total campaign spend."
                )

        result["executive_summary"] = summary

        # ====================================================
        # ACTIONABLE RECOMMENDATIONS
        # ====================================================

        recommendations = []

        # Recommendation 1
        if (
            "zero_conversion_campaigns"
            in result
            and
            len(
                result[
                    "zero_conversion_campaigns"
                ]
            ) > 0
        ):

            recommendations.append({

                "priority": "high",

                "category": "conversion",

                "issue": (
                    "Campaigns are spending money "
                    "without producing conversions."
                ),

                "action": (
                    "Review targeting, creative, landing pages, "
                    "offer quality, tracking configuration, "
                    "and audience fit before increasing budgets."
                ),

                "affected_campaigns": len(
                    result[
                        "zero_conversion_campaigns"
                    ]
                ),

            })

        # Recommendation 2
        if (
            "underperforming_campaigns"
            in result
            and
            len(
                result[
                    "underperforming_campaigns"
                ]
            ) > 0
        ):

            recommendations.append({

                "priority": "high",

                "category": "roi",

                "issue": (
                    "Some campaigns have ROAS below "
                    "the break-even level."
                ),

                "action": (
                    "Review or optimize campaigns with "
                    "ROAS below 1 before allocating additional budget."
                ),

                "affected_campaigns": len(
                    result[
                        "underperforming_campaigns"
                    ]
                ),

            })

        # Recommendation 3
        if (
            "budget_concentration"
            in result
        ):

            concentration = result[
                "budget_concentration"
            ][
                "top_5_spend_percentage"
            ]

            if concentration >= 60:

                recommendations.append({

                    "priority": "medium",

                    "category": "budget_risk",

                    "issue": (
                        "Marketing spend is highly concentrated "
                        "in a small number of campaigns."
                    ),

                    "action": (
                        "Monitor the largest campaigns closely "
                        "and maintain diversified campaign coverage "
                        "to reduce budget concentration risk."
                    ),

                    "top_5_spend_percentage": (
                        concentration
                    ),

                })

        # Recommendation 4
        if (
            "best_channel_by_roas"
            in result
        ):

            best_channel = result[
                "best_channel_by_roas"
            ]

            recommendations.append({

                "priority": "medium",

                "category": "channel_optimization",

                "issue": (
                    "A channel is currently generating "
                    "strong relative return."
                ),

                "action": (
                    f"Investigate whether {best_channel['channel']} "
                    "can support additional qualified budget while "
                    "monitoring marginal performance."
                ),

                "channel": best_channel[
                    "channel"
                ],

                "roas": best_channel[
                    "roas"
                ],

            })

        # Recommendation 5
        if (
            "anomalies" in result
            and
            result["anomalies"]
        ):

            recommendations.append({

                "priority": "medium",

                "category": "data_and_performance",

                "issue": (
                    "Performance anomalies were detected."
                ),

                "action": (
                    "Investigate unusually high spend or "
                    "conversion performance to determine whether "
                    "the results represent genuine performance "
                    "or tracking/data-quality issues."
                ),

                "anomaly_count": len(
                    result["anomalies"]
                ),

            })

        # Data-quality recommendation
        if data_quality["warnings"]:

            recommendations.append({

                "priority": "medium",

                "category": "data_quality",

                "issue": (
                    "Campaign data contains quality limitations."
                ),

                "action": (
                    "Correct missing, duplicated, negative, "
                    "or inconsistent campaign records before "
                    "making major budget decisions."
                ),

                "warnings": data_quality[
                    "warnings"
                ],

            })

        result["recommendations"] = (
            recommendations
        )

        # ====================================================
        # REQUESTED PROMPT / ANALYSIS CONTEXT
        # ====================================================

        if prompt:

            result["analysis_context"] = (
                str(prompt).strip()
            )

        # ====================================================
        # FINAL JSON SAFETY
        # ====================================================

        return _json_safe(
            result
        )

    except Exception as e:

        return {
            "error": str(e),
            "agent": AGENT_NAME,
        }