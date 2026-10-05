 from __future__ import annotations

import json
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# SALES PIPELINE ANALYST SYSTEM PROMPT
# ============================================================

SALES_PIPELINE_PROMPT = """
You are an elite sales pipeline intelligence analyst focused on
pipeline health, sales execution, conversion, deal velocity,
forecasting, risk detection, and revenue opportunity.

MISSION
- Analyze pipeline health across stages, representatives, and segments.
- Compute conversion, win/loss, pipeline, and forecast metrics.
- Detect stalled opportunities, bottlenecks, aging deals, and forecast risk.
- Identify high-value opportunities requiring immediate attention.
- Evaluate sales representative performance using available evidence.
- Recommend practical actions prioritized by business impact.
- Never invent data that is not present in the dataset.

ANALYSIS PRINCIPLES
1. Use actual dataset values as the primary evidence.
2. Clearly distinguish observed metrics from estimates and forecasts.
3. Never claim forecast certainty.
4. When probability is unavailable, use transparent stage-based assumptions.
5. Do not treat pipeline value as revenue unless the dataset explicitly
   identifies it as revenue or closed/won value.
6. Consider both deal count and monetary value.
7. Consider deal aging and close dates when available.
8. Identify concentration risk.
9. Separate pipeline quality problems from sales execution problems.
10. Avoid unfair comparisons between representatives with very different
    numbers of opportunities.
11. Surface data-quality limitations that could affect conclusions.
12. Recommendations must be specific, measurable, and actionable.

FORECASTING
- Use explicit probability when available.
- Otherwise use transparent stage-based probability assumptions.
- Provide conservative, base, and best-case scenarios when possible.
- Label estimated forecasts clearly.

RISK DETECTION
Look for:
- overdue deals
- stale deals
- high-value stalled deals
- large deals with low probability
- excessive pipeline concentration
- weak conversion
- unusual stage concentration
- missing close dates
- missing values
- abnormal deal sizes
- inactive or inconsistent stages
- excessive dependence on a single representative

OUTPUT
Produce:
- executive summary
- pipeline health
- stage intelligence
- financial intelligence
- forecasting
- deal aging
- sales velocity
- win/loss intelligence
- representative intelligence
- concentration analysis
- anomalies
- bottlenecks
- risks
- opportunities
- prioritized recommendations
- confidence and data-quality assessment
"""


AGENT_NAME = "sales-pipeline"


# ============================================================
# GENERIC HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return float(value)

        number = float(value)

        if pd.isna(number):
            return default

        return number
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        number = int(value)

        return number
    except Exception:
        return default


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def _normalize_stage(value: Any) -> str:
    text = _clean_text(value)

    if not text:
        return "Unknown"

    return text


def _json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy objects into JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Find a column case-insensitively while preserving the real column name.
    """
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def _numeric_series(
    df: pd.DataFrame,
    column: Optional[str],
) -> pd.Series:
    if not column or column not in df.columns:
        return pd.Series(dtype="float64")

    values = (
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(values, errors="coerce")


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    # Remove completely empty columns.
    working = working.dropna(axis=1, how="all")

    # Normalize column names without destroying original data.
    working.columns = [
        str(column).strip()
        for column in working.columns
    ]

    # Remove completely empty rows.
    working = working.dropna(axis=0, how="all").reset_index(drop=True)

    return working


# ============================================================
# COLUMN DISCOVERY
# ============================================================

def _discover_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "deal": _find_column(
            df,
            [
                "Deal",
                "deal",
                "Deal Name",
                "deal_name",
                "Opportunity",
                "Opportunity Name",
                "opportunity",
            ],
        ),
        "stage": _find_column(
            df,
            [
                "Stage",
                "stage",
                "Pipeline Stage",
                "pipeline_stage",
            ],
        ),
        "amount": _find_column(
            df,
            [
                "Amount",
                "amount",
                "Deal Amount",
                "deal_amount",
                "Value",
                "value",
                "Pipeline Value",
                "pipeline_value",
            ],
        ),
        "rep": _find_column(
            df,
            [
                "Rep",
                "rep",
                "Sales Rep",
                "sales_rep",
                "Salesperson",
                "Owner",
                "owner",
                "Account Executive",
            ],
        ),
        "close_date": _find_column(
            df,
            [
                "Close_Date",
                "Close Date",
                "close_date",
                "close date",
                "Expected Close",
                "Expected Close Date",
                "expected_close_date",
            ],
        ),
        "created_date": _find_column(
            df,
            [
                "Created_Date",
                "Created Date",
                "created_date",
                "created date",
                "Lead Created",
                "Opportunity Created",
            ],
        ),
        "probability": _find_column(
            df,
            [
                "Probability",
                "probability",
                "Win Probability",
                "win_probability",
                "Probability %",
                "probability_pct",
            ],
        ),
        "status": _find_column(
            df,
            [
                "Status",
                "status",
                "Deal Status",
                "deal_status",
                "Outcome",
                "outcome",
            ],
        ),
        "deal_id": _find_column(
            df,
            [
                "Deal_ID",
                "Deal ID",
                "deal_id",
                "ID",
                "id",
                "Opportunity ID",
                "opportunity_id",
            ],
        ),
    }


# ============================================================
# STAGE PROBABILITY
# ============================================================

def _stage_probability(stage: str) -> float:
    """
    Transparent fallback probabilities.

    These are estimates only and should not be interpreted as
    historical company-specific win probabilities.
    """
    text = _clean_text(stage).lower()

    if not text:
        return 0.0

    if any(
        keyword in text
        for keyword in [
            "closed won",
            "closed-won",
            "won",
            "customer",
        ]
    ):
        return 1.0

    if any(
        keyword in text
        for keyword in [
            "closed lost",
            "closed-lost",
            "lost",
            "dead",
        ]
    ):
        return 0.0

    if any(
        keyword in text
        for keyword in [
            "proposal",
            "negotiation",
            "contract",
            "verbal",
            "commit",
        ]
    ):
        return 0.70

    if any(
        keyword in text
        for keyword in [
            "demo",
            "evaluation",
            "presentation",
            "qualified",
        ]
    ):
        return 0.45

    if any(
        keyword in text
        for keyword in [
            "discovery",
            "meeting",
            "contact",
        ]
    ):
        return 0.25

    if any(
        keyword in text
        for keyword in [
            "lead",
            "prospect",
            "new",
        ]
    ):
        return 0.10

    return 0.20


def _classify_stage(stage: str) -> str:
    text = _clean_text(stage).lower()

    if any(
        keyword in text
        for keyword in [
            "closed won",
            "closed-won",
            "won",
            "customer",
        ]
    ):
        return "won"

    if any(
        keyword in text
        for keyword in [
            "closed lost",
            "closed-lost",
            "lost",
            "dead",
        ]
    ):
        return "lost"

    return "open"


# ============================================================
# DATA QUALITY
# ============================================================

def _calculate_data_quality(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    rows = len(df)

    missing_cells = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    amount_column = columns.get("amount")

    invalid_amounts = 0
    negative_amounts = 0

    if amount_column:
        amounts = _numeric_series(df, amount_column)

        invalid_amounts = int(
            df[amount_column].notna().sum()
            - amounts.notna().sum()
        )

        negative_amounts = int(
            (amounts < 0).sum()
        )

    completeness_pct = 100.0

    total_cells = rows * max(len(df.columns), 1)

    if total_cells > 0:
        completeness_pct = round(
            ((total_cells - missing_cells) / total_cells) * 100,
            2,
        )

    quality_score = 100.0

    quality_score -= min(
        completeness_pct < 90,
        1,
    ) * 10

    if rows > 0:
        quality_score -= min(
            duplicate_rows / rows * 100,
            15,
        )

    quality_score -= min(invalid_amounts * 2, 10)
    quality_score -= min(negative_amounts * 2, 10)

    quality_score = max(
        0.0,
        min(100.0, quality_score),
    )

    return {
        "quality_score": round(quality_score, 2),
        "completeness_pct": completeness_pct,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "invalid_amounts": invalid_amounts,
        "negative_amounts": negative_amounts,
        "available_columns": list(df.columns),
    }


# ============================================================
# STAGE INTELLIGENCE
# ============================================================

def _calculate_stage_intelligence(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    stage_column = columns.get("stage")

    if not stage_column:
        return {
            "available": False,
            "reason": "Stage column not available.",
        }

    amount_column = columns.get("amount")

    amounts = (
        _numeric_series(df, amount_column)
        if amount_column
        else pd.Series(
            0.0,
            index=df.index,
        )
    )

    stage_analysis: Dict[str, Any] = {}

    for stage, group in df.groupby(
        stage_column,
        dropna=False,
    ):
        stage_name = _normalize_stage(stage)

        indexes = group.index

        stage_amounts = amounts.loc[indexes]

        valid_amounts = stage_amounts.dropna()

        count = len(group)

        total_amount = (
            float(valid_amounts.sum())
            if len(valid_amounts)
            else 0.0
        )

        probability = _stage_probability(stage_name)

        weighted_amount = total_amount * probability

        stage_analysis[stage_name] = {
            "deals": count,
            "total_amount": round(total_amount, 2),
            "average_amount": round(
                float(valid_amounts.mean()),
                2,
            ) if len(valid_amounts) else 0.0,
            "stage_type": _classify_stage(stage_name),
            "estimated_probability_pct": round(
                probability * 100,
                2,
            ),
            "estimated_weighted_value": round(
                weighted_amount,
                2,
            ),
        }

    ordered = sorted(
        stage_analysis.items(),
        key=lambda item: item[1]["total_amount"],
        reverse=True,
    )

    return {
        "available": True,
        "stages": dict(ordered),
        "stage_count": len(stage_analysis),
    }


# ============================================================
# FINANCIAL INTELLIGENCE
# ============================================================

def _calculate_financial_intelligence(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    amount_column = columns.get("amount")

    if not amount_column:
        return {
            "available": False,
            "reason": "Amount column not available.",
        }

    amounts = _numeric_series(
        df,
        amount_column,
    )

    valid = amounts.dropna()

    if len(valid) == 0:
        return {
            "available": False,
            "reason": "No valid numeric deal amounts found.",
        }

    total = float(valid.sum())

    average = float(valid.mean())

    median = float(valid.median())

    largest = float(valid.max())

    smallest = float(valid.min())

    std = float(valid.std()) if len(valid) > 1 else 0.0

    high_value_threshold = max(
        median * 2,
        average,
    )

    high_value_count = int(
        (valid >= high_value_threshold).sum()
    )

    high_value_amount = float(
        valid[valid >= high_value_threshold].sum()
    )

    return {
        "available": True,
        "total_pipeline": round(total, 2),
        "average_deal_size": round(average, 2),
        "median_deal_size": round(median, 2),
        "largest_deal": round(largest, 2),
        "smallest_deal": round(smallest, 2),
        "deal_count_with_amount": int(len(valid)),
        "amount_std_dev": round(std, 2),
        "high_value_threshold": round(
            high_value_threshold,
            2,
        ),
        "high_value_deal_count": high_value_count,
        "high_value_pipeline_amount": round(
            high_value_amount,
            2,
        ),
    }


# ============================================================
# FORECASTING
# ============================================================

def _calculate_forecast(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    amount_column = columns.get("amount")

    if not amount_column:
        return {
            "available": False,
            "reason": "Amount column not available.",
        }

    amounts = _numeric_series(
        df,
        amount_column,
    )

    probability_column = columns.get("probability")

    if probability_column:
        probabilities = _numeric_series(
            df,
            probability_column,
        )

        probabilities = probabilities.where(
            probabilities <= 1,
            probabilities / 100,
        )

        probabilities = probabilities.clip(
            lower=0,
            upper=1,
        )

        probability_source = "explicit_probability"
    else:
        stage_column = columns.get("stage")

        if stage_column:
            probabilities = df[stage_column].apply(
                lambda stage: _stage_probability(
                    _normalize_stage(stage)
                )
            )

            probability_source = "stage_based_estimate"
        else:
            probabilities = pd.Series(
                0.20,
                index=df.index,
            )

            probability_source = "generic_estimate"

    valid_mask = (
        amounts.notna()
        & probabilities.notna()
    )

    if not valid_mask.any():
        return {
            "available": False,
            "reason": "Insufficient numeric data for forecast.",
        }

    valid_amounts = amounts[valid_mask]

    valid_probabilities = probabilities[valid_mask]

    weighted_pipeline = float(
        (
            valid_amounts
            * valid_probabilities
        ).sum()
    )

    total_pipeline = float(
        valid_amounts.sum()
    )

    open_pipeline = 0.0

    stage_column = columns.get("stage")

    if stage_column:
        for index in valid_amounts.index:
            stage = _normalize_stage(
                df.loc[index, stage_column]
            )

            if _classify_stage(stage) == "open":
                open_pipeline += float(
                    valid_amounts.loc[index]
                )
    else:
        open_pipeline = total_pipeline

    conservative = weighted_pipeline * 0.80

    best_case = weighted_pipeline + (
        open_pipeline * 0.25
    )

    return {
        "available": True,
        "probability_source": probability_source,
        "total_pipeline": round(
            total_pipeline,
            2,
        ),
        "weighted_pipeline": round(
            weighted_pipeline,
            2,
        ),
        "conservative_forecast": round(
            conservative,
            2,
        ),
        "base_forecast": round(
            weighted_pipeline,
            2,
        ),
        "best_case_forecast": round(
            best_case,
            2,
        ),
        "weighted_coverage_pct": round(
            (
                weighted_pipeline
                / total_pipeline
                * 100
            )
            if total_pipeline > 0
            else 0,
            2,
        ),
        "assumption_note": (
            "Forecast is probability-weighted. "
            "Explicit deal probabilities are used when available; "
            "otherwise stage-based estimates are used."
        ),
    }


# ============================================================
# DEAL AGING
# ============================================================

def _calculate_deal_aging(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    created_column = columns.get("created_date")

    if not created_column:
        return {
            "available": False,
            "reason": "Created date column not available.",
        }

    dates = pd.to_datetime(
        df[created_column],
        errors="coerce",
    )

    valid_dates = dates.dropna()

    if len(valid_dates) == 0:
        return {
            "available": False,
            "reason": "No valid created dates found.",
        }

    reference_date = pd.Timestamp.now()

    ages = (
        reference_date
        - valid_dates
    ).dt.days.clip(lower=0)

    buckets = {
        "0_30_days": int(
            (ages <= 30).sum()
        ),
        "31_60_days": int(
            ((ages > 30) & (ages <= 60)).sum()
        ),
        "61_90_days": int(
            ((ages > 60) & (ages <= 90)).sum()
        ),
        "91_180_days": int(
            ((ages > 90) & (ages <= 180)).sum()
        ),
        "181_plus_days": int(
            (ages > 180).sum()
        ),
    }

    return {
        "available": True,
        "reference_date": reference_date.isoformat(),
        "average_age_days": round(
            float(ages.mean()),
            2,
        ),
        "median_age_days": round(
            float(ages.median()),
            2,
        ),
        "oldest_age_days": int(
            ages.max()
        ),
        "aging_buckets": buckets,
        "stale_over_90_days": int(
            (ages > 90).sum()
        ),
        "stale_over_180_days": int(
            (ages > 180).sum()
        ),
    }


# ============================================================
# CLOSE DATE INTELLIGENCE
# ============================================================

def _calculate_close_date_intelligence(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    close_column = columns.get("close_date")

    if not close_column:
        return {
            "available": False,
            "reason": "Close date column not available.",
        }

    dates = pd.to_datetime(
        df[close_column],
        errors="coerce",
    )

    valid = dates.dropna()

    if len(valid) == 0:
        return {
            "available": False,
            "reason": "No valid close dates found.",
        }

    today = pd.Timestamp.now().normalize()

    overdue_mask = valid < today

    next_7 = valid <= (
        today
        + pd.Timedelta(days=7)
    )

    next_30 = valid <= (
        today
        + pd.Timedelta(days=30)
    )

    future_mask = valid >= today

    return {
        "available": True,
        "valid_close_dates": int(len(valid)),
        "overdue_deals": int(
            overdue_mask.sum()
        ),
        "closing_within_7_days": int(
            (next_7 & future_mask).sum()
        ),
        "closing_within_30_days": int(
            (next_30 & future_mask).sum()
        ),
        "missing_close_dates": int(
            dates.isna().sum()
        ),
    }


# ============================================================
# WIN / LOSS INTELLIGENCE
# ============================================================

def _calculate_win_loss(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    status_column = columns.get("status")
    stage_column = columns.get("stage")

    if status_column:
        source = df[status_column].astype(str)
        source_name = status_column
    elif stage_column:
        source = df[stage_column].astype(str)
        source_name = stage_column
    else:
        return {
            "available": False,
            "reason": "Status or stage column not available.",
        }

    normalized = source.str.strip().str.lower()

    won_mask = normalized.str.contains(
        r"won|closed won|closed-won|customer",
        regex=True,
        na=False,
    )

    lost_mask = normalized.str.contains(
        r"lost|closed lost|closed-lost|dead",
        regex=True,
        na=False,
    )

    won = int(won_mask.sum())
    lost = int(lost_mask.sum())

    decided = won + lost

    return {
        "available": True,
        "source_column": source_name,
        "won_deals": won,
        "lost_deals": lost,
        "decided_deals": decided,
        "win_rate_pct": _safe_pct(
            won,
            decided,
        ),
        "loss_rate_pct": _safe_pct(
            lost,
            decided,
        ),
        "undecided_deals": int(
            len(df) - decided
        ),
    }


# ============================================================
# SALES VELOCITY
# ============================================================

def _calculate_sales_velocity(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    created_column = columns.get("created_date")
    close_column = columns.get("close_date")

    if not created_column or not close_column:
        return {
            "available": False,
            "reason": (
                "Created date and close date are required "
                "for sales velocity."
            ),
        }

    created = pd.to_datetime(
        df[created_column],
        errors="coerce",
    )

    close = pd.to_datetime(
        df[close_column],
        errors="coerce",
    )

    duration = (
        close - created
    ).dt.days

    duration = duration[
        duration >= 0
    ].dropna()

    if len(duration) == 0:
        return {
            "available": False,
            "reason": "No valid sales cycle durations found.",
        }

    return {
        "available": True,
        "average_sales_cycle_days": round(
            float(duration.mean()),
            2,
        ),
        "median_sales_cycle_days": round(
            float(duration.median()),
            2,
        ),
        "fastest_sales_cycle_days": int(
            duration.min()
        ),
        "slowest_sales_cycle_days": int(
            duration.max()
        ),
        "deals_with_cycle_data": int(
            len(duration)
        ),
    }


# ============================================================
# REPRESENTATIVE INTELLIGENCE
# ============================================================

def _calculate_rep_intelligence(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    rep_column = columns.get("rep")
    amount_column = columns.get("amount")

    if not rep_column:
        return {
            "available": False,
            "reason": "Rep column not available.",
        }

    amount_values = (
        _numeric_series(
            df,
            amount_column,
        )
        if amount_column
        else pd.Series(
            0.0,
            index=df.index,
        )
    )

    results: Dict[str, Any] = {}

    for rep, group in df.groupby(
        rep_column,
        dropna=False,
    ):
        rep_name = _clean_text(rep) or "Unknown"

        group_amounts = amount_values.loc[
            group.index
        ].dropna()

        total_amount = float(
            group_amounts.sum()
        )

        average_amount = (
            float(group_amounts.mean())
            if len(group_amounts)
            else 0.0
        )

        results[rep_name] = {
            "deals": int(len(group)),
            "total_amount": round(
                total_amount,
                2,
            ),
            "average_deal_size": round(
                average_amount,
                2,
            ),
        }

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["total_amount"],
        reverse=True,
    )

    return {
        "available": True,
        "representatives": dict(ranking),
        "representative_count": len(results),
        "top_rep_by_pipeline": (
            ranking[0][0]
            if ranking
            else None
        ),
    }


# ============================================================
# CONCENTRATION ANALYSIS
# ============================================================

def _calculate_concentration(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> Dict[str, Any]:

    amount_column = columns.get("amount")

    if not amount_column:
        return {
            "available": False,
            "reason": "Amount column not available.",
        }

    amounts = _numeric_series(
        df,
        amount_column,
    ).dropna()

    if len(amounts) == 0:
        return {
            "available": False,
            "reason": "No valid amounts found.",
        }

    total = float(amounts.sum())

    largest = float(amounts.max())

    top_3 = float(
        amounts.nlargest(
            min(3, len(amounts))
        ).sum()
    )

    top_10 = float(
        amounts.nlargest(
            min(10, len(amounts))
        ).sum()
    )

    return {
        "available": True,
        "largest_deal_share_pct": _safe_pct(
            largest,
            total,
        ),
        "top_3_deals_share_pct": _safe_pct(
            top_3,
            total,
        ),
        "top_10_deals_share_pct": _safe_pct(
            top_10,
            total,
        ),
        "concentration_risk": (
            "high"
            if _safe_pct(largest, total) >= 30
            else "moderate"
            if _safe_pct(largest, total) >= 15
            else "low"
        ),
    }


# ============================================================
# ANOMALIES
# ============================================================

def _detect_anomalies(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:

    anomalies: List[Dict[str, Any]] = []

    amount_column = columns.get("amount")
    stage_column = columns.get("stage")
    close_column = columns.get("close_date")

    if amount_column:
        amounts = _numeric_series(
            df,
            amount_column,
        )

        valid = amounts.dropna()

        if len(valid) >= 3:
            threshold = float(
                valid.mean()
                + (
                    2
                    * valid.std()
                )
            )

            high_value_count = int(
                (valid > threshold).sum()
            )

            if high_value_count > 0:
                anomalies.append(
                    {
                        "type": "unusually_large_deals",
                        "count": high_value_count,
                        "threshold": round(
                            threshold,
                            2,
                        ),
                        "severity": "medium",
                    }
                )

        zero_amounts = int(
            (valid <= 0).sum()
        )

        if zero_amounts > 0:
            anomalies.append(
                {
                    "type": "zero_or_negative_amounts",
                    "count": zero_amounts,
                    "severity": "high",
                }
            )

    if stage_column:
        stage_counts = (
            df[stage_column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        if len(stage_counts) > 0:
            largest_stage = stage_counts.iloc[0]

            largest_stage_share = _safe_pct(
                int(largest_stage),
                len(df),
            )

            if largest_stage_share >= 60:
                anomalies.append(
                    {
                        "type": "stage_concentration",
                        "stage": str(
                            stage_counts.index[0]
                        ),
                        "share_pct": largest_stage_share,
                        "severity": "medium",
                    }
                )

    if close_column:
        dates = pd.to_datetime(
            df[close_column],
            errors="coerce",
        )

        today = pd.Timestamp.now().normalize()

        overdue = int(
            (dates < today).sum()
        )

        if overdue > 0:
            anomalies.append(
                {
                    "type": "overdue_close_dates",
                    "count": overdue,
                    "severity": "high",
                }
            )

    return anomalies


# ============================================================
# BOTTLENECK DETECTION
# ============================================================

def _identify_bottlenecks(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:

    bottlenecks: List[Dict[str, Any]] = []

    stage_column = columns.get("stage")

    if not stage_column:
        return bottlenecks

    stage_counts = (
        df[stage_column]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    if len(stage_counts) == 0:
        return bottlenecks

    total = len(df)

    for stage, count in stage_counts.items():
        share = _safe_pct(
            int(count),
            total,
        )

        if share >= 40:
            bottlenecks.append(
                {
                    "stage": str(stage),
                    "deal_count": int(count),
                    "pipeline_share_pct": share,
                    "reason": (
                        "High concentration of deals in this stage. "
                        "Review progression, aging, and conversion."
                    ),
                }
            )

    return bottlenecks


# ============================================================
# RISK ENGINE
# ============================================================

def _identify_risks(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
    forecast: Dict[str, Any],
    win_loss: Dict[str, Any],
    concentration: Dict[str, Any],
) -> List[Dict[str, Any]]:

    risks: List[Dict[str, Any]] = []

    if win_loss.get("available"):
        win_rate = win_loss.get(
            "win_rate_pct",
            0,
        )

        if (
            win_loss.get("decided_deals", 0) >= 5
            and win_rate < 25
        ):
            risks.append(
                {
                    "type": "low_win_rate",
                    "severity": "high",
                    "metric": win_rate,
                    "message": (
                        "Historical win rate is low and may "
                        "reduce forecast reliability."
                    ),
                }
            )

    if concentration.get("available"):
        if concentration.get(
            "concentration_risk"
        ) == "high":
            risks.append(
                {
                    "type": "pipeline_concentration",
                    "severity": "high",
                    "metric": concentration.get(
                        "largest_deal_share_pct",
                        0,
                    ),
                    "message": (
                        "A large portion of pipeline depends "
                        "on a small number of deals."
                    ),
                }
            )

    close_column = columns.get("close_date")

    if close_column:
        dates = pd.to_datetime(
            df[close_column],
            errors="coerce",
        )

        today = pd.Timestamp.now().normalize()

        overdue = int(
            (dates < today).sum()
        )

        if overdue > 0:
            risks.append(
                {
                    "type": "overdue_pipeline",
                    "severity": "high",
                    "count": overdue,
                    "message": (
                        "Deals have close dates in the past "
                        "and require pipeline cleanup or acceleration."
                    ),
                }
            )

    if forecast.get("available"):
        total = forecast.get(
            "total_pipeline",
            0,
        )

        weighted = forecast.get(
            "weighted_pipeline",
            0,
        )

        if total > 0:
            weighted_pct = (
                weighted
                / total
                * 100
            )

            if weighted_pct < 30:
                risks.append(
                    {
                        "type": "low_weighted_pipeline",
                        "severity": "medium",
                        "weighted_pipeline_pct": round(
                            weighted_pct,
                            2,
                        ),
                        "message": (
                            "A relatively small portion of pipeline "
                            "is probability-weighted toward expected value."
                        ),
                    }
                )

    return risks


# ============================================================
# OPPORTUNITY ENGINE
# ============================================================

def _identify_opportunities(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:

    opportunities: List[Dict[str, Any]] = []

    amount_column = columns.get("amount")
    stage_column = columns.get("stage")

    if not amount_column:
        return opportunities

    amounts = _numeric_series(
        df,
        amount_column,
    )

    valid = amounts.dropna()

    if len(valid) == 0:
        return opportunities

    threshold = max(
        float(valid.median() * 2),
        float(valid.mean()),
    )

    high_value = int(
        (valid >= threshold).sum()
    )

    if high_value > 0:
        opportunities.append(
            {
                "type": "high_value_opportunities",
                "count": high_value,
                "threshold": round(
                    threshold,
                    2,
                ),
                "action": (
                    "Prioritize executive sponsorship, "
                    "deal strategy, and close-plan reviews."
                ),
            }
        )

    if stage_column:
        stage_series = (
            df[stage_column]
            .fillna("Unknown")
            .astype(str)
        )

        open_mask = ~stage_series.str.lower().str.contains(
            r"won|lost|closed",
            regex=True,
            na=False,
        )

        open_amount = float(
            amounts[
                open_mask
            ].dropna().sum()
        )

        if open_amount > 0:
            opportunities.append(
                {
                    "type": "open_pipeline",
                    "amount": round(
                        open_amount,
                        2,
                    ),
                    "action": (
                        "Focus sales capacity on open opportunities "
                        "with strong stage progression."
                    ),
                }
            )

    return opportunities


# ============================================================
# RECOMMENDATIONS
# ============================================================

def _generate_recommendations(
    results: Dict[str, Any],
) -> List[Dict[str, Any]]:

    recommendations: List[Dict[str, Any]] = []

    risks = results.get(
        "risks",
        [],
    )

    anomalies = results.get(
        "anomalies",
        [],
    )

    bottlenecks = results.get(
        "bottlenecks",
        [],
    )

    forecast = results.get(
        "forecast",
        {},
    )

    for risk in risks[:5]:
        recommendations.append(
            {
                "priority": "high"
                if risk.get("severity") == "high"
                else "medium",
                "action": risk.get(
                    "message",
                    "Review pipeline risk.",
                ),
                "reason": risk.get(
                    "type",
                    "pipeline_risk",
                ),
            }
        )

    for bottleneck in bottlenecks[:3]:
        recommendations.append(
            {
                "priority": "high",
                "action": (
                    f"Review the {bottleneck.get('stage')} stage "
                    "for stalled opportunities and progression blockers."
                ),
                "reason": "stage_bottleneck",
            }
        )

    for anomaly in anomalies[:3]:
        recommendations.append(
            {
                "priority": "medium",
                "action": (
                    f"Investigate {anomaly.get('type')} "
                    "before using the pipeline for executive forecasting."
                ),
                "reason": anomaly.get(
                    "type",
                    "pipeline_anomaly",
                ),
            }
        )

    if forecast.get("available"):
        recommendations.append(
            {
                "priority": "medium",
                "action": (
                    "Maintain conservative, base, and best-case "
                    "forecast scenarios rather than relying on "
                    "a single pipeline number."
                ),
                "reason": "forecast_governance",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "medium",
                "action": (
                    "Continue monitoring stage progression, "
                    "deal aging, close dates, and conversion."
                ),
                "reason": "pipeline_monitoring",
            }
        )

    return recommendations[:10]


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def _build_executive_summary(
    results: Dict[str, Any],
) -> Dict[str, Any]:

    financial = results.get(
        "financial_intelligence",
        {},
    )

    forecast = results.get(
        "forecast",
        {},
    )

    win_loss = results.get(
        "win_loss",
        {},
    )

    aging = results.get(
        "deal_aging",
        {},
    )

    risks = results.get(
        "risks",
        [],
    )

    opportunities = results.get(
        "opportunities",
        [],
    )

    total_pipeline = financial.get(
        "total_pipeline",
        0,
    )

    weighted = forecast.get(
        "weighted_pipeline",
        0,
    )

    win_rate = win_loss.get(
        "win_rate_pct",
        0,
    )

    stale = aging.get(
        "stale_over_90_days",
        0,
    )

    return {
        "total_deals": results.get(
            "rows",
            0,
        ),
        "total_pipeline": total_pipeline,
        "weighted_pipeline": weighted,
        "win_rate_pct": win_rate,
        "stale_over_90_days": stale,
        "risk_count": len(risks),
        "opportunity_count": len(opportunities),
        "pipeline_health": (
            "at_risk"
            if len(risks) >= 3
            else "needs_attention"
            if len(risks) > 0
            else "healthy"
        ),
    }


# ============================================================
# MAIN ANALYSIS ENGINE
# ============================================================

def _analyze_dataframe(
    df: pd.DataFrame,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:

    df = _prepare_dataframe(df)

    columns = _discover_columns(df)

    results: Dict[str, Any] = {
        # ----------------------------------------------------
        # ORIGINAL OUTPUTS - PRESERVED
        # ----------------------------------------------------
        "rows": len(df),
        "columns": df.columns.tolist(),
    }

    # ========================================================
    # ORIGINAL STAGE ANALYSIS
    # ========================================================

    stage_column = columns.get("stage")

    if stage_column:
        stage_counts = (
            df[stage_column]
            .value_counts()
            .to_dict()
        )

        results["stage_distribution"] = stage_counts

    # ========================================================
    # ORIGINAL FINANCIAL ANALYSIS
    # ========================================================

    amount_column = columns.get("amount")

    if amount_column:
        amounts = _numeric_series(
            df,
            amount_column,
        )

        results["financial_summary"] = {
            "total_pipeline": float(
                amounts.sum()
            ),
            "average_deal_size": float(
                amounts.mean()
            )
            if amounts.notna().any()
            else 0.0,
            "deal_count": len(df),
            "largest_deal": float(
                amounts.max()
            )
            if amounts.notna().any()
            else 0.0,
        }

    # ========================================================
    # ORIGINAL REP ANALYSIS
    # ========================================================

    rep_column = columns.get("rep")

    if rep_column and amount_column:
        rep_summary: Dict[str, Any] = {}

        for rep in df[rep_column].unique():
            rep_df = df[
                df[rep_column] == rep
            ]

            rep_amounts = _numeric_series(
                rep_df,
                amount_column,
            )

            rep_summary[
                str(rep)
            ] = {
                "deals": len(rep_df),
                "total_amount": float(
                    rep_amounts.sum()
                ),
            }

        results["rep_performance"] = rep_summary

    # ========================================================
    # ADVANCED INTELLIGENCE
    # ========================================================

    results["agent"] = AGENT_NAME

    results["analysis_context"] = (
        prompt or ""
    )

    results["system_prompt"] = SALES_PIPELINE_PROMPT

    results["detected_columns"] = columns

    results["data_quality"] = (
        _calculate_data_quality(
            df,
            columns,
        )
    )

    results["stage_intelligence"] = (
        _calculate_stage_intelligence(
            df,
            columns,
        )
    )

    results["financial_intelligence"] = (
        _calculate_financial_intelligence(
            df,
            columns,
        )
    )

    results["forecast"] = (
        _calculate_forecast(
            df,
            columns,
        )
    )

    results["deal_aging"] = (
        _calculate_deal_aging(
            df,
            columns,
        )
    )

    results["close_date_intelligence"] = (
        _calculate_close_date_intelligence(
            df,
            columns,
        )
    )

    results["win_loss"] = (
        _calculate_win_loss(
            df,
            columns,
        )
    )

    results["sales_velocity"] = (
        _calculate_sales_velocity(
            df,
            columns,
        )
    )

    results["rep_intelligence"] = (
        _calculate_rep_intelligence(
            df,
            columns,
        )
    )

    results["pipeline_concentration"] = (
        _calculate_concentration(
            df,
            columns,
        )
    )

    results["anomalies"] = (
        _detect_anomalies(
            df,
            columns,
        )
    )

    results["bottlenecks"] = (
        _identify_bottlenecks(
            df,
            columns,
        )
    )

    results["risks"] = (
        _identify_risks(
            df,
            columns,
            results["forecast"],
            results["win_loss"],
            results["pipeline_concentration"],
        )
    )

    results["opportunities"] = (
        _identify_opportunities(
            df,
            columns,
        )
    )

    results["recommendations"] = (
        _generate_recommendations(
            results,
        )
    )

    results["executive_summary"] = (
        _build_executive_summary(
            results,
        )
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    quality_score = results[
        "data_quality"
    ].get(
        "quality_score",
        0,
    )

    confidence = quality_score

    if not amount_column:
        confidence -= 20

    if not stage_column:
        confidence -= 10

    if not columns.get("close_date"):
        confidence -= 5

    confidence = max(
        0.0,
        min(100.0, confidence),
    )

    results["confidence"] = {
        "score": round(
            confidence,
            2,
        ),
        "level": (
            "high"
            if confidence >= 80
            else "medium"
            if confidence >= 60
            else "low"
        ),
        "explanation": (
            "Confidence reflects data completeness, "
            "available pipeline fields, and data quality. "
            "Forecast confidence may be lower when probabilities "
            "are estimated from stages."
        ),
    }

    return _json_safe(results)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def analyze_sales_pipeline(
    csv_content: str,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze sales pipeline CSV.

    Supported common columns include:

    Deal
    Stage
    Amount
    Rep
    Close_Date
    Created_Date
    Probability
    Status
    Deal_ID

    The original output fields are preserved while advanced
    intelligence is added.
    """

    try:
        if not csv_content:
            return {
                "error": (
                    "sales-pipeline requires CSV content."
                ),
                "agent": AGENT_NAME,
            }

        df = pd.read_csv(
            StringIO(csv_content)
        )

        return _analyze_dataframe(
            df,
            prompt,
        )

    except pd.errors.EmptyDataError:
        return {
            "error": "The supplied CSV is empty.",
            "agent": AGENT_NAME,
        }

    except pd.errors.ParserError as exc:
        return {
            "error": (
                f"Unable to parse CSV: {str(exc)}"
            ),
            "agent": AGENT_NAME,
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "agent": AGENT_NAME,
        }


# ============================================================
# PUBLIC RUNNER
# ============================================================

def run_sales_pipeline_analyst(
    csv_content: Optional[str],
    table_json: Optional[List[Dict[str, Any]]],
    prompt: str,
) -> Dict[str, Any]:
    """
    Main agent entry point.

    Supports either:
    - CSV input
    - table_json input

    Both inputs use the same analysis engine.
    """

    try:
        if csv_content:
            return analyze_sales_pipeline(
                csv_content,
                prompt,
            )

        if table_json is not None:
            if not isinstance(
                table_json,
                list,
            ):
                return {
                    "error": (
                        "table_json must be a list of objects."
                    ),
                    "agent": AGENT_NAME,
                }

            df = pd.DataFrame(
                table_json
            )

            return _analyze_dataframe(
                df,
                prompt,
            )

        return {
            "error": (
                "sales-pipeline requires CSV "
                "or table JSON input."
            ),
            "agent": AGENT_NAME,
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "agent": AGENT_NAME,
        }