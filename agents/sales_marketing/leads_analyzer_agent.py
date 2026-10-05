 """
Leads Analyzer Agent
Advanced AI Sales Lead Intelligence, Qualification, Conversion,
Pipeline, Source, Aging, Risk, Opportunity, and Lead Prioritization Analyzer.

Preserves the original lead-analysis functionality while adding:
- Data quality analysis
- Status normalization
- Funnel analysis
- Stage conversion analysis
- Source performance analysis
- Lead value segmentation
- Lead scoring
- Lead prioritization
- Lead aging and stale-lead detection
- Conversion analysis
- Pipeline concentration
- Company intelligence
- Anomaly detection
- Trend analysis
- Forecasting when sufficient date data exists
- Risk and opportunity detection
- Executive summaries
- Actionable recommendations
- Confidence scoring
"""

import json
import math
import re
from io import StringIO
from typing import Optional, List, Dict, Any

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

AGENT_NAME = "leads-analyzer"

DEFAULT_TOP_N = 10
DEFAULT_HIGH_VALUE_PERCENTILE = 75
DEFAULT_STALE_DAYS = 60
DEFAULT_CRITICAL_AGE_DAYS = 90

EPSILON = 0.0000001


# ============================================================
# SAFE / GENERAL HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    Prevents NaN and infinity from entering JSON output.
    """
    try:
        if value is None:
            return default

        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer.
    """
    try:
        if value is None:
            return default

        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return default

        return int(number)

    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0
) -> float:
    """
    Safe division helper.
    """
    num = _safe_float(numerator)
    den = _safe_float(denominator)

    if abs(den) <= EPSILON:
        return default

    return num / den


def _clean_text(value: Any) -> str:
    """
    Normalize text values for analysis.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def _normalize_status(value: Any) -> str:
    """
    Normalize common lead-status variations.

    This does not overwrite the original status column.
    """
    value = _clean_text(value).lower()

    if not value:
        return "unknown"

    normalized = re.sub(r"[_\-]+", " ", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    mapping = {
        "new": "new",
        "new lead": "new",
        "prospect": "prospect",
        "prospecting": "prospect",
        "contacted": "contacted",
        "contact": "contacted",
        "attempted contact": "contacted",
        "qualified": "qualified",
        "mql": "qualified",
        "sql": "qualified",
        "sales qualified": "qualified",
        "opportunity": "opportunity",
        "open opportunity": "opportunity",
        "converted": "converted",
        "won": "converted",
        "closed won": "converted",
        "customer": "converted",
        "lost": "lost",
        "closed lost": "lost",
        "disqualified": "disqualified",
        "unqualified": "disqualified",
        "dead": "disqualified",
        "rejected": "disqualified",
        "nurture": "nurture",
        "nurturing": "nurture",
        "working": "working",
        "in progress": "working",
    }

    return mapping.get(normalized, normalized)


def _normalize_stage(value: Any) -> str:
    """
    Normalize pipeline-stage labels without modifying original data.
    """
    value = _clean_text(value).lower()

    if not value:
        return "unknown"

    normalized = re.sub(r"[_\-]+", " ", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def _json_safe(value: Any) -> Any:
    """
    Recursively convert values into JSON-safe Python values.
    """
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, (pd.Series,)):
        return _json_safe(value.to_dict())

    if isinstance(value, (pd.DataFrame,)):
        return _json_safe(value.to_dict(orient="records"))

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0

    return value


def _round_number(value: Any, digits: int = 2) -> float:
    return round(_safe_float(value), digits)


# ============================================================
# COLUMN DISCOVERY
# ============================================================

def _find_column(
    df: pd.DataFrame,
    candidates: List[str]
) -> Optional[str]:
    """
    Find a column using exact or normalized names.
    """
    if df.empty and len(df.columns) == 0:
        return None

    normalized_columns = {
        re.sub(r"[\s_\-]+", "", str(column).strip().lower()): column
        for column in df.columns
    }

    for candidate in candidates:
        normalized_candidate = re.sub(
            r"[\s_\-]+",
            "",
            candidate.strip().lower()
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "date",
            "lead_date",
            "created_at",
            "created_date",
            "created",
            "lead_created_at",
            "lead_created_date",
            "inquiry_date",
            "contact_date",
            "updated_at",
            "updated_date",
            "conversion_date",
            "converted_at",
        ],
    )


def _find_value_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "value",
            "lead_value",
            "deal_value",
            "pipeline_value",
            "amount",
            "revenue",
            "opportunity_value",
        ],
    )


def _find_status_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "status",
            "lead_status",
            "lead_stage",
            "lifecycle_stage",
        ],
    )


def _find_source_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "source",
            "lead_source",
            "marketing_source",
            "acquisition_source",
            "channel",
            "origin",
        ],
    )


def _find_company_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "company",
            "company_name",
            "account",
            "account_name",
            "organization",
        ],
    )


def _find_stage_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "stage",
            "pipeline_stage",
            "sales_stage",
            "deal_stage",
        ],
    )


def _find_age_column(df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        df,
        [
            "age_days",
            "lead_age_days",
            "days_old",
            "days_in_pipeline",
            "days_open",
            "age",
        ],
    )


# ============================================================
# DATA QUALITY
# ============================================================

def _analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze lead dataset quality without modifying the original data.
    """
    expected_columns = [
        "lead_id",
        "source",
        "status",
        "company",
        "value",
        "stage",
        "age_days",
    ]

    available_columns = [
        str(column)
        for column in df.columns
    ]

    normalized_available = {
        re.sub(r"[\s_\-]+", "", column.lower())
        for column in available_columns
    }

    missing_expected = []

    for column in expected_columns:
        normalized = re.sub(
            r"[\s_\-]+",
            "",
            column.lower()
        )

        if normalized not in normalized_available:
            missing_expected.append(column)

    missing_values = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            missing_values[str(column)] = {
                "missing_count": missing_count,
                "missing_pct": _round_number(
                    _safe_divide(
                        missing_count,
                        len(df)
                    ) * 100
                ),
            }

    duplicate_rows = int(df.duplicated().sum())

    duplicate_lead_ids = 0

    lead_id_column = _find_column(
        df,
        [
            "lead_id",
            "id",
            "leadid",
        ],
    )

    if lead_id_column:
        duplicate_lead_ids = int(
            df[lead_id_column]
            .duplicated(keep=False)
            .sum()
        )

    negative_values = {}

    value_column = _find_value_column(df)

    if value_column:
        numeric_values = pd.to_numeric(
            df[value_column],
            errors="coerce"
        )

        negative_count = int(
            (numeric_values < 0).sum()
        )

        if negative_count > 0:
            negative_values[value_column] = negative_count

    age_column = _find_age_column(df)

    invalid_age_count = 0

    if age_column:
        numeric_age = pd.to_numeric(
            df[age_column],
            errors="coerce"
        )

        invalid_age_count = int(
            (numeric_age < 0).sum()
        )

    warnings = []

    if missing_expected:
        warnings.append(
            "Some expected lead-analysis columns are missing."
        )

    if duplicate_rows > 0:
        warnings.append(
            "Duplicate rows were detected."
        )

    if duplicate_lead_ids > 0:
        warnings.append(
            "Duplicate lead IDs were detected."
        )

    if negative_values:
        warnings.append(
            "Negative lead-value records were detected."
        )

    if invalid_age_count > 0:
        warnings.append(
            "Negative lead-age values were detected."
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "available_columns": available_columns,
        "missing_expected_columns": missing_expected,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_lead_ids": duplicate_lead_ids,
        "negative_values": negative_values,
        "invalid_age_count": invalid_age_count,
        "warnings": warnings,
        "quality_status": (
            "good"
            if not warnings
            else "review_required"
        ),
    }


# ============================================================
# ORIGINAL STATUS ANALYSIS
# ============================================================

def _analyze_statuses(
    df: pd.DataFrame,
    status_column: Optional[str]
) -> Dict[str, Any]:
    """
    Preserve original status analysis while adding normalized
    status intelligence.
    """
    if not status_column:
        return {}

    original_status = (
        df[status_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    normalized_status = original_status.apply(
        _normalize_status
    )

    status_dist = original_status.value_counts().to_dict()

    normalized_dist = (
        normalized_status
        .value_counts()
        .to_dict()
    )

    qualified = int(
        (normalized_status == "qualified").sum()
    )

    converted = int(
        (normalized_status == "converted").sum()
    )

    lost = int(
        (normalized_status == "lost").sum()
    )

    disqualified = int(
        (normalized_status == "disqualified").sum()
    )

    active_statuses = [
        "new",
        "prospect",
        "contacted",
        "working",
        "qualified",
        "opportunity",
        "nurture",
    ]

    active_leads = int(
        normalized_status.isin(active_statuses).sum()
    )

    return {
        "status_distribution": status_dist,
        "normalized_status_distribution": normalized_dist,
        "qualified_leads": qualified,
        "converted_leads": converted,
        "lost_leads": lost,
        "disqualified_leads": disqualified,
        "active_leads": active_leads,
    }


# ============================================================
# FUNNEL ANALYSIS
# ============================================================

def _analyze_funnel(
    df: pd.DataFrame,
    status_column: Optional[str],
    stage_column: Optional[str]
) -> Dict[str, Any]:
    """
    Analyze lead funnel progression.
    """
    if not status_column and not stage_column:
        return {}

    funnel_source = (
        status_column
        if status_column
        else stage_column
    )

    values = df[funnel_source].fillna("").astype(str)

    if funnel_source == status_column:
        values = values.apply(_normalize_status)
    else:
        values = values.apply(_normalize_stage)

    counts = values.value_counts()

    funnel = []

    ordered_statuses = [
        "new",
        "prospect",
        "contacted",
        "working",
        "qualified",
        "opportunity",
        "converted",
        "lost",
        "disqualified",
        "nurture",
    ]

    seen = set()

    for status in ordered_statuses:
        if status in counts.index:
            count = int(counts.get(status, 0))

            funnel.append(
                {
                    "stage": status,
                    "leads": count,
                    "pct_of_total": _round_number(
                        _safe_divide(
                            count,
                            len(df)
                        ) * 100
                    ),
                }
            )

            seen.add(status)

    for status, count in counts.items():
        if status not in seen:
            count = int(count)

            funnel.append(
                {
                    "stage": str(status),
                    "leads": count,
                    "pct_of_total": _round_number(
                        _safe_divide(
                            count,
                            len(df)
                        ) * 100
                    ),
                }
            )

    converted_count = int(
        (values == "converted").sum()
    )

    qualified_count = int(
        (values == "qualified").sum()
    )

    return {
        "funnel": funnel,
        "qualified_to_converted_rate_pct": _round_number(
            _safe_divide(
                converted_count,
                qualified_count
            ) * 100
        ),
        "overall_funnel_conversion_pct": _round_number(
            _safe_divide(
                converted_count,
                len(df)
            ) * 100
        ),
    }


# ============================================================
# VALUE ANALYSIS
# ============================================================

def _analyze_value(
    df: pd.DataFrame,
    value_column: Optional[str]
) -> Dict[str, Any]:
    """
    Advanced lead-value intelligence.
    """
    if not value_column:
        return {}

    values = pd.to_numeric(
        df[value_column],
        errors="coerce"
    ).fillna(0)

    if len(values) == 0:
        return {}

    total = values.sum()
    average = values.mean()
    median = values.median()
    maximum = values.max()

    percentile_75 = values.quantile(0.75)
    percentile_90 = values.quantile(0.90)

    high_value_mask = values >= percentile_75
    very_high_value_mask = values >= percentile_90

    return {
        "total_pipeline_value": _round_number(total),
        "avg_lead_value": _round_number(average),
        "max_lead_value": _round_number(maximum),
        "median_lead_value": _round_number(median),

        "value_percentiles": {
            "p75": _round_number(percentile_75),
            "p90": _round_number(percentile_90),
        },

        "high_value_leads": int(
            high_value_mask.sum()
        ),

        "very_high_value_leads": int(
            very_high_value_mask.sum()
        ),

        "high_value_pipeline": _round_number(
            values[high_value_mask].sum()
        ),

        "high_value_pipeline_pct": _round_number(
            _safe_divide(
                values[high_value_mask].sum(),
                total
            ) * 100
        ),

        "zero_value_leads": int(
            (values <= 0).sum()
        ),

        "positive_value_leads": int(
            (values > 0).sum()
        ),
    }


# ============================================================
# SOURCE INTELLIGENCE
# ============================================================

def _analyze_sources(
    df: pd.DataFrame,
    source_column: Optional[str],
    status_column: Optional[str],
    value_column: Optional[str]
) -> Dict[str, Any]:
    """
    Advanced source performance analysis.
    """
    if not source_column:
        return {}

    working = df.copy()

    working["_analysis_source"] = (
        working[source_column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    if status_column:
        working["_analysis_status"] = (
            working[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )
    else:
        working["_analysis_status"] = "unknown"

    if value_column:
        working["_analysis_value"] = pd.to_numeric(
            working[value_column],
            errors="coerce"
        ).fillna(0)
    else:
        working["_analysis_value"] = 0.0

    records = []

    for source, group in working.groupby(
        "_analysis_source",
        dropna=False
    ):
        total = len(group)

        converted = int(
            (group["_analysis_status"] == "converted").sum()
        )

        qualified = int(
            (group["_analysis_status"] == "qualified").sum()
        )

        lost = int(
            (group["_analysis_status"] == "lost").sum()
        )

        total_value = group["_analysis_value"].sum()

        conversion_rate = (
            _safe_divide(
                converted,
                total
            ) * 100
        )

        qualification_rate = (
            _safe_divide(
                qualified,
                total
            ) * 100
        )

        loss_rate = (
            _safe_divide(
                lost,
                total
            ) * 100
        )

        records.append(
            {
                "source": str(source),
                "leads": int(total),
                "qualified": qualified,
                "converted": converted,
                "lost": lost,
                "total_value": _round_number(total_value),
                "avg_value": _round_number(
                    _safe_divide(
                        total_value,
                        total
                    )
                ),
                "qualification_rate_pct": _round_number(
                    qualification_rate
                ),
                "conversion_rate_pct": _round_number(
                    conversion_rate
                ),
                "loss_rate_pct": _round_number(
                    loss_rate
                ),
                "value_per_lead": _round_number(
                    _safe_divide(
                        total_value,
                        total
                    )
                ),
            }
        )

    records.sort(
        key=lambda item: (
            item["conversion_rate_pct"],
            item["total_value"]
        ),
        reverse=True,
    )

    source_performance = {
        record["source"]: {
            "converted": record["converted"],
            "avg_value": record["avg_value"],
        }
        for record in records
    }

    result = {
        "source_performance": records,
        "source_ranking": records[:DEFAULT_TOP_N],
        "source_efficiency": source_performance,
    }

    if records:
        result["best_source_by_conversion"] = records[0]

        best_value_source = max(
            records,
            key=lambda item: item["total_value"]
        )

        result["best_source_by_pipeline_value"] = (
            best_value_source
        )

        best_quality_source = max(
            records,
            key=lambda item: (
                item["qualification_rate_pct"],
                item["conversion_rate_pct"]
            )
        )

        result["best_source_by_lead_quality"] = (
            best_quality_source
        )

    return result


# ============================================================
# STAGE ANALYSIS
# ============================================================

def _analyze_stages(
    df: pd.DataFrame,
    stage_column: Optional[str],
    status_column: Optional[str],
    value_column: Optional[str]
) -> Dict[str, Any]:
    """
    Advanced pipeline-stage analysis.
    """
    if not stage_column:
        return {}

    working = df.copy()

    working["_analysis_stage"] = (
        working[stage_column]
        .fillna("Unknown")
        .astype(str)
        .apply(_normalize_stage)
    )

    if status_column:
        working["_analysis_status"] = (
            working[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )
    else:
        working["_analysis_status"] = "unknown"

    if value_column:
        working["_analysis_value"] = pd.to_numeric(
            working[value_column],
            errors="coerce"
        ).fillna(0)
    else:
        working["_analysis_value"] = 0.0

    records = []

    for stage, group in working.groupby(
        "_analysis_stage",
        dropna=False
    ):
        leads = len(group)

        converted = int(
            (group["_analysis_status"] == "converted").sum()
        )

        total_value = group["_analysis_value"].sum()

        records.append(
            {
                "stage": str(stage),
                "leads": int(leads),
                "converted": converted,
                "total_value": _round_number(total_value),
                "avg_value": _round_number(
                    _safe_divide(
                        total_value,
                        leads
                    )
                ),
                "conversion_rate_pct": _round_number(
                    _safe_divide(
                        converted,
                        leads
                    ) * 100
                ),
            }
        )

    records.sort(
        key=lambda item: item["leads"],
        reverse=True
    )

    return {
        "stage_performance": records,
        "largest_stage_by_volume": (
            records[0]
            if records
            else None
        ),
        "highest_value_stage": (
            max(
                records,
                key=lambda item: item["total_value"]
            )
            if records
            else None
        ),
    }


# ============================================================
# LEAD AGING
# ============================================================

def _analyze_aging(
    df: pd.DataFrame,
    age_column: Optional[str],
    status_column: Optional[str],
    value_column: Optional[str]
) -> Dict[str, Any]:
    """
    Advanced lead-aging and stale-lead analysis.
    """
    if not age_column:
        return {}

    ages = pd.to_numeric(
        df[age_column],
        errors="coerce"
    ).fillna(0)

    if status_column:
        statuses = (
            df[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )
    else:
        statuses = pd.Series(
            ["unknown"] * len(df),
            index=df.index
        )

    if value_column:
        values = pd.to_numeric(
            df[value_column],
            errors="coerce"
        ).fillna(0)
    else:
        values = pd.Series(
            [0.0] * len(df),
            index=df.index
        )

    over_30 = ages > 30
    over_60 = ages > 60
    over_90 = ages > 90

    stale_mask = (
        ages >= DEFAULT_STALE_DAYS
    ) & (
        ~statuses.isin(
            [
                "converted",
                "lost",
                "disqualified",
            ]
        )
    )

    critical_mask = (
        ages >= DEFAULT_CRITICAL_AGE_DAYS
    ) & (
        ~statuses.isin(
            [
                "converted",
                "lost",
                "disqualified",
            ]
        )
    )

    stale_value = values[stale_mask].sum()

    return {
        "avg_age_days": _round_number(
            ages.mean()
        ),
        "median_age_days": _round_number(
            ages.median()
        ),
        "max_age_days": _round_number(
            ages.max()
        ),

        "leads_over_30_days": int(
            over_30.sum()
        ),

        "leads_over_60_days": int(
            over_60.sum()
        ),

        "leads_over_90_days": int(
            over_90.sum()
        ),

        "stale_active_leads": int(
            stale_mask.sum()
        ),

        "critical_age_leads": int(
            critical_mask.sum()
        ),

        "stale_pipeline_value": _round_number(
            stale_value
        ),

        "stale_pipeline_pct": _round_number(
            _safe_divide(
                stale_value,
                values.sum()
            ) * 100
        ),
    }


# ============================================================
# LEAD SCORING
# ============================================================

def _calculate_lead_scores(
    df: pd.DataFrame,
    status_column: Optional[str],
    value_column: Optional[str],
    age_column: Optional[str],
    source_column: Optional[str]
) -> Dict[str, Any]:
    """
    Calculate deterministic lead-priority scores.

    The score is an analytical prioritization score, not a
    replacement for a trained ML model.
    """
    if len(df) == 0:
        return {}

    working = df.copy()

    if status_column:
        statuses = (
            working[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )
    else:
        statuses = pd.Series(
            ["unknown"] * len(working),
            index=working.index
        )

    if value_column:
        values = pd.to_numeric(
            working[value_column],
            errors="coerce"
        ).fillna(0)
    else:
        values = pd.Series(
            [0.0] * len(working),
            index=working.index
        )

    if age_column:
        ages = pd.to_numeric(
            working[age_column],
            errors="coerce"
        ).fillna(0)
    else:
        ages = pd.Series(
            [0.0] * len(working),
            index=working.index
        )

    max_value = values.max()

    if max_value > 0:
        value_score = (
            values / max_value
        ) * 40
    else:
        value_score = pd.Series(
            [0.0] * len(working),
            index=working.index
        )

    status_points = {
        "converted": 100,
        "opportunity": 90,
        "qualified": 80,
        "working": 65,
        "contacted": 55,
        "prospect": 45,
        "new": 40,
        "nurture": 30,
        "lost": 10,
        "disqualified": 5,
        "unknown": 20,
    }

    status_score = statuses.map(
        lambda status: status_points.get(
            status,
            20
        )
    )

    age_score = (
        20
        - (
            ages.clip(
                lower=0,
                upper=120
            ) / 120 * 20
        )
    )

    age_score = age_score.clip(
        lower=0,
        upper=20
    )

    conversion_bonus = statuses.apply(
        lambda status: (
            20
            if status in [
                "qualified",
                "opportunity"
            ]
            else 0
        )
    )

    scores = (
        value_score
        + status_score * 0.4
        + age_score
        + conversion_bonus
    )

    scores = scores.clip(
        lower=0,
        upper=100
    )

    priority = scores.apply(
        lambda score: (
            "critical"
            if score >= 85
            else (
                "high"
                if score >= 70
                else (
                    "medium"
                    if score >= 50
                    else "low"
                )
            )
        )
    )

    return {
        "lead_scoring": {
            "method": (
                "Deterministic analytical score using "
                "lead value, status, age, and qualification signals."
            ),
            "score_range": "0-100",
            "critical_threshold": 85,
            "high_threshold": 70,
            "medium_threshold": 50,
        },

        "priority_distribution": (
            priority.value_counts()
            .to_dict()
        ),

        "average_lead_score": _round_number(
            scores.mean()
        ),

        "top_priority_leads": [
            {
                "row_index": int(index),
                "score": _round_number(
                    scores.loc[index]
                ),
                "priority": str(
                    priority.loc[index]
                ),
                "status": str(
                    statuses.loc[index]
                ),
                "value": _round_number(
                    values.loc[index]
                ),
                "age_days": _round_number(
                    ages.loc[index]
                ),
            }
            for index in scores.sort_values(
                ascending=False
            ).head(DEFAULT_TOP_N).index
        ],
    }


# ============================================================
# CONVERSION METRICS
# ============================================================

def _analyze_conversion(
    df: pd.DataFrame,
    status_column: Optional[str]
) -> Dict[str, Any]:
    """
    Advanced conversion metrics.
    """
    if not status_column:
        return {}

    statuses = (
        df[status_column]
        .fillna("")
        .astype(str)
        .apply(_normalize_status)
    )

    total = len(df)

    converted = int(
        (statuses == "converted").sum()
    )

    qualified = int(
        (statuses == "qualified").sum()
    )

    opportunity = int(
        (statuses == "opportunity").sum()
    )

    lost = int(
        (statuses == "lost").sum()
    )

    closed_records = converted + lost

    result = {
        "conversion_rate_pct": _round_number(
            _safe_divide(
                converted,
                total
            ) * 100
        ),

        "lead_quality_index": _round_number(
            _safe_divide(
                converted,
                total
            ) * 100
        ),

        "qualification_rate_pct": _round_number(
            _safe_divide(
                qualified,
                total
            ) * 100
        ),

        "opportunity_rate_pct": _round_number(
            _safe_divide(
                opportunity,
                total
            ) * 100
        ),

        "loss_rate_pct": _round_number(
            _safe_divide(
                lost,
                total
            ) * 100
        ),

        "closed_lead_win_rate_pct": _round_number(
            _safe_divide(
                converted,
                closed_records
            ) * 100
        ),
    }

    return result


# ============================================================
# COMPANY INTELLIGENCE
# ============================================================

def _analyze_companies(
    df: pd.DataFrame,
    company_column: Optional[str],
    status_column: Optional[str],
    value_column: Optional[str]
) -> Dict[str, Any]:
    """
    Company/account-level lead intelligence.
    """
    if not company_column:
        return {}

    working = df.copy()

    working["_company"] = (
        working[company_column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    if status_column:
        working["_status"] = (
            working[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )
    else:
        working["_status"] = "unknown"

    if value_column:
        working["_value"] = pd.to_numeric(
            working[value_column],
            errors="coerce"
        ).fillna(0)
    else:
        working["_value"] = 0.0

    records = []

    for company, group in working.groupby(
        "_company",
        dropna=False
    ):
        lead_count = len(group)

        converted = int(
            (group["_status"] == "converted").sum()
        )

        qualified = int(
            (group["_status"] == "qualified").sum()
        )

        total_value = group["_value"].sum()

        records.append(
            {
                "company": str(company),
                "leads": int(lead_count),
                "qualified": qualified,
                "converted": converted,
                "total_value": _round_number(total_value),
                "avg_value": _round_number(
                    _safe_divide(
                        total_value,
                        lead_count
                    )
                ),
                "conversion_rate_pct": _round_number(
                    _safe_divide(
                        converted,
                        lead_count
                    ) * 100
                ),
            }
        )

    records.sort(
        key=lambda item: (
            item["total_value"],
            item["converted"]
        ),
        reverse=True
    )

    return {
        "company_performance": records[:DEFAULT_TOP_N],
        "company_count": len(records),
    }


# ============================================================
# PIPELINE CONCENTRATION
# ============================================================

def _analyze_concentration(
    df: pd.DataFrame,
    value_column: Optional[str],
    source_column: Optional[str]
) -> Dict[str, Any]:
    """
    Identify concentration risk in pipeline value.
    """
    result = {}

    if value_column:
        values = pd.to_numeric(
            df[value_column],
            errors="coerce"
        ).fillna(0)

        total_value = values.sum()

        sorted_values = values.sort_values(
            ascending=False
        )

        top_5_value = sorted_values.head(5).sum()

        result["top_5_lead_value"] = _round_number(
            top_5_value
        )

        result["top_5_lead_value_pct"] = _round_number(
            _safe_divide(
                top_5_value,
                total_value
            ) * 100
        )

    if source_column and value_column:
        grouped = (
            df.assign(
                _source=df[source_column]
                .fillna("Unknown")
                .astype(str)
            )
            .assign(
                _value=pd.to_numeric(
                    df[value_column],
                    errors="coerce"
                ).fillna(0)
            )
            .groupby("_source")["_value"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        total = grouped.sum()

        if len(grouped) > 0:
            top_source_value = grouped.iloc[0]

            result["top_source_pipeline_value"] = (
                _round_number(top_source_value)
            )

            result["top_source_pipeline_pct"] = (
                _round_number(
                    _safe_divide(
                        top_source_value,
                        total
                    ) * 100
                )
            )

            result["source_concentration_risk"] = (
                "high"
                if _safe_divide(
                    top_source_value,
                    total
                ) >= 0.60
                else (
                    "medium"
                    if _safe_divide(
                        top_source_value,
                        total
                    ) >= 0.40
                    else "low"
                )
            )

    return result


# ============================================================
# ANOMALY DETECTION
# ============================================================

def _detect_anomalies(
    df: pd.DataFrame,
    value_column: Optional[str],
    age_column: Optional[str],
    status_column: Optional[str]
) -> Dict[str, Any]:
    """
    Detect unusual lead records.
    """
    anomalies = []

    if len(df) == 0:
        return {
            "count": 0,
            "items": [],
        }

    if value_column:
        values = pd.to_numeric(
            df[value_column],
            errors="coerce"
        ).fillna(0)

        mean_value = values.mean()
        std_value = values.std()

        high_value_threshold = (
            mean_value + 2 * std_value
            if std_value > 0
            else mean_value
        )

        high_value_indices = values[
            values > high_value_threshold
        ].index

        for index in high_value_indices[:20]:
            anomalies.append(
                {
                    "type": "unusually_high_lead_value",
                    "row_index": int(index),
                    "value": _round_number(
                        values.loc[index]
                    ),
                    "threshold": _round_number(
                        high_value_threshold
                    ),
                }
            )

    if age_column:
        ages = pd.to_numeric(
            df[age_column],
            errors="coerce"
        ).fillna(0)

        high_age_threshold = max(
            DEFAULT_CRITICAL_AGE_DAYS,
            ages.mean() + 2 * ages.std()
            if ages.std() > 0
            else DEFAULT_CRITICAL_AGE_DAYS
        )

        high_age_indices = ages[
            ages > high_age_threshold
        ].index

        for index in high_age_indices[:20]:
            anomalies.append(
                {
                    "type": "extremely_aged_lead",
                    "row_index": int(index),
                    "age_days": _round_number(
                        ages.loc[index]
                    ),
                    "threshold": _round_number(
                        high_age_threshold
                    ),
                }
            )

    if (
        status_column
        and value_column
    ):
        statuses = (
            df[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )

        values = pd.to_numeric(
            df[value_column],
            errors="coerce"
        ).fillna(0)

        suspicious_mask = (
            values > 0
        ) & (
            statuses.isin(
                [
                    "lost",
                    "disqualified"
                ]
            )
        )

        for index in values[
            suspicious_mask
        ].index[:20]:
            anomalies.append(
                {
                    "type": "value_attached_to_lost_or_disqualified_lead",
                    "row_index": int(index),
                    "value": _round_number(
                        values.loc[index]
                    ),
                    "status": str(
                        statuses.loc[index]
                    ),
                }
            )

    return {
        "count": len(anomalies),
        "items": anomalies,
    }


# ============================================================
# TREND ANALYSIS
# ============================================================

def _analyze_trends(
    df: pd.DataFrame,
    date_column: Optional[str],
    status_column: Optional[str],
    value_column: Optional[str]
) -> Dict[str, Any]:
    """
    Analyze monthly lead trends when a usable date column exists.
    """
    if not date_column:
        return {
            "available": False,
            "reason": "No usable date column was detected.",
        }

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    valid_mask = dates.notna()

    if valid_mask.sum() < 2:
        return {
            "available": False,
            "reason": "Not enough valid dates for trend analysis.",
        }

    working = df.loc[
        valid_mask
    ].copy()

    working["_date"] = dates.loc[
        valid_mask
    ]

    working["_month"] = (
        working["_date"]
        .dt.to_period("M")
        .astype(str)
    )

    if status_column:
        working["_status"] = (
            working[status_column]
            .fillna("")
            .astype(str)
            .apply(_normalize_status)
        )
    else:
        working["_status"] = "unknown"

    if value_column:
        working["_value"] = pd.to_numeric(
            working[value_column],
            errors="coerce"
        ).fillna(0)
    else:
        working["_value"] = 0.0

    records = []

    for month, group in working.groupby(
        "_month",
        sort=True
    ):
        leads = len(group)

        converted = int(
            (group["_status"] == "converted").sum()
        )

        total_value = group["_value"].sum()

        records.append(
            {
                "month": str(month),
                "leads": int(leads),
                "converted": converted,
                "conversion_rate_pct": _round_number(
                    _safe_divide(
                        converted,
                        leads
                    ) * 100
                ),
                "pipeline_value": _round_number(
                    total_value
                ),
            }
        )

    result = {
        "available": True,
        "monthly_trends": records,
    }

    if len(records) >= 2:
        previous = records[-2]
        current = records[-1]

        lead_change_pct = (
            _safe_divide(
                current["leads"] - previous["leads"],
                previous["leads"]
            ) * 100
        )

        conversion_change = (
            current["conversion_rate_pct"]
            - previous["conversion_rate_pct"]
        )

        value_change_pct = (
            _safe_divide(
                current["pipeline_value"]
                - previous["pipeline_value"],
                previous["pipeline_value"]
            ) * 100
        )

        result["latest_month_change"] = {
            "lead_volume_change_pct": _round_number(
                lead_change_pct
            ),
            "conversion_rate_change_pct_points": _round_number(
                conversion_change
            ),
            "pipeline_value_change_pct": _round_number(
                value_change_pct
            ),
        }

    return result


# ============================================================
# FORECAST
# ============================================================

def _forecast_conversions(
    trends: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simple transparent forecast based on recent observed
    conversion performance.

    No forecast is produced when insufficient historical data
    exists.
    """
    monthly = trends.get(
        "monthly_trends",
        []
    )

    if len(monthly) < 3:
        return {
            "available": False,
            "reason": (
                "At least 3 months of valid lead history "
                "are recommended for forecasting."
            ),
        }

    recent = monthly[-3:]

    avg_leads = sum(
        item["leads"]
        for item in recent
    ) / len(recent)

    avg_conversion_rate = sum(
        item["conversion_rate_pct"]
        for item in recent
    ) / len(recent)

    projected_conversions = (
        avg_leads
        * avg_conversion_rate
        / 100
    )

    return {
        "available": True,
        "method": (
            "Three-month average lead volume multiplied "
            "by three-month average conversion rate."
        ),
        "recent_months_used": len(recent),
        "average_monthly_leads": _round_number(
            avg_leads
        ),
        "average_conversion_rate_pct": _round_number(
            avg_conversion_rate
        ),
        "projected_monthly_conversions": _round_number(
            projected_conversions
        ),
    }


# ============================================================
# RISK DETECTION
# ============================================================

def _detect_risks(
    result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert analytical findings into explicit business risks.
    """
    risks = []

    aging = result.get(
        "lead_aging",
        {}
    )

    if aging.get(
        "critical_age_leads",
        0
    ) > 0:
        risks.append(
            {
                "priority": "high",
                "type": "lead_aging",
                "message": (
                    f"{aging['critical_age_leads']} active leads "
                    "are older than the critical aging threshold."
                ),
            }
        )

    concentration = result.get(
        "pipeline_concentration",
        {}
    )

    if concentration.get(
        "source_concentration_risk"
    ) == "high":
        risks.append(
            {
                "priority": "high",
                "type": "source_concentration",
                "message": (
                    "A large share of pipeline value depends "
                    "on one lead source."
                ),
            }
        )

    anomalies = result.get(
        "anomalies",
        {}
    )

    if anomalies.get(
        "count",
        0
    ) > 0:
        risks.append(
            {
                "priority": "medium",
                "type": "data_or_pipeline_anomalies",
                "message": (
                    f"{anomalies['count']} unusual lead records "
                    "require review."
                ),
            }
        )

    data_quality = result.get(
        "data_quality",
        {}
    )

    if data_quality.get(
        "warnings"
    ):
        risks.append(
            {
                "priority": "medium",
                "type": "data_quality",
                "message": (
                    "Lead data quality issues may affect "
                    "analysis reliability."
                ),
            }
        )

    conversion = result.get(
        "conversion_metrics",
        {}
    )

    if (
        conversion
        and conversion.get(
            "conversion_rate_pct",
            0
        ) < 5
    ):
        risks.append(
            {
                "priority": "high",
                "type": "low_conversion",
                "message": (
                    "Overall lead conversion is below 5%; "
                    "funnel quality or sales follow-up should "
                    "be investigated."
                ),
            }
        )

    return risks


# ============================================================
# OPPORTUNITY DETECTION
# ============================================================

def _detect_opportunities(
    result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Identify positive business opportunities.
    """
    opportunities = []

    best_source = result.get(
        "best_source_by_conversion"
    )

    if best_source:
        opportunities.append(
            {
                "priority": "medium",
                "type": "high_performing_source",
                "source": best_source.get(
                    "source"
                ),
                "message": (
                    "Consider increasing attention or investment "
                    "in the strongest converting source while "
                    "monitoring incremental performance."
                ),
            }
        )

    value_analysis = result.get(
        "value_analysis",
        {}
    )

    if (
        value_analysis.get(
            "high_value_leads",
            0
        ) > 0
    ):
        opportunities.append(
            {
                "priority": "high",
                "type": "high_value_leads",
                "message": (
                    "High-value leads represent an opportunity "
                    "for prioritized sales follow-up."
                ),
                "high_value_leads": value_analysis.get(
                    "high_value_leads",
                    0
                ),
                "high_value_pipeline": value_analysis.get(
                    "high_value_pipeline",
                    0
                ),
            }
        )

    lead_scoring = result.get(
        "lead_scoring",
        {}
    )

    priority_distribution = result.get(
        "lead_priority_distribution",
        {}
    )

    if priority_distribution.get(
        "high",
        0
    ) > 0 or priority_distribution.get(
        "critical",
        0
    ) > 0:
        opportunities.append(
            {
                "priority": "high",
                "type": "sales_prioritization",
                "message": (
                    "High-priority leads have been identified "
                    "for immediate sales attention."
                ),
                "priority_distribution": priority_distribution,
            }
        )

    return opportunities


# ============================================================
# RECOMMENDATIONS
# ============================================================

def _generate_recommendations(
    result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate prioritized, actionable recommendations.
    """
    recommendations = []

    aging = result.get(
        "lead_aging",
        {}
    )

    stale_count = aging.get(
        "stale_active_leads",
        0
    )

    if stale_count > 0:
        recommendations.append(
            {
                "priority": "high",
                "area": "lead_aging",
                "action": (
                    "Review and re-engage stale active leads."
                ),
                "reason": (
                    f"{stale_count} active leads have been "
                    f"stale for at least {DEFAULT_STALE_DAYS} days."
                ),
            }
        )

    value_analysis = result.get(
        "value_analysis",
        {}
    )

    high_value = value_analysis.get(
        "high_value_leads",
        0
    )

    if high_value > 0:
        recommendations.append(
            {
                "priority": "high",
                "area": "lead_prioritization",
                "action": (
                    "Prioritize high-value leads for "
                    "personalized sales follow-up."
                ),
                "reason": (
                    f"{high_value} leads are in the top "
                    "value segment."
                ),
            }
        )

    conversion = result.get(
        "conversion_metrics",
        {}
    )

    conversion_rate = conversion.get(
        "conversion_rate_pct",
        0
    )

    if conversion_rate < 5:
        recommendations.append(
            {
                "priority": "high",
                "area": "conversion",
                "action": (
                    "Investigate lead qualification, follow-up "
                    "speed, and funnel-stage leakage."
                ),
                "reason": (
                    f"Overall conversion is "
                    f"{conversion_rate}%."
                ),
            }
        )

    source = result.get(
        "best_source_by_conversion"
    )

    if source:
        recommendations.append(
            {
                "priority": "medium",
                "area": "lead_source",
                "action": (
                    "Evaluate whether the best-performing "
                    "lead source can be scaled."
                ),
                "reason": (
                    f"{source.get('source')} has the strongest "
                    "observed conversion rate."
                ),
            }
        )

    concentration = result.get(
        "pipeline_concentration",
        {}
    )

    if concentration.get(
        "source_concentration_risk"
    ) == "high":
        recommendations.append(
            {
                "priority": "high",
                "area": "source_diversification",
                "action": (
                    "Diversify lead acquisition sources "
                    "to reduce concentration risk."
                ),
                "reason": (
                    "One source represents a disproportionately "
                    "large share of pipeline value."
                ),
            }
        )

    if result.get(
        "data_quality",
        {}
    ).get(
        "warnings"
    ):
        recommendations.append(
            {
                "priority": "medium",
                "area": "data_quality",
                "action": (
                    "Clean and standardize lead data before "
                    "making high-impact decisions."
                ),
                "reason": (
                    "The dataset contains quality warnings."
                ),
            }
        )

    if result.get(
        "anomalies",
        {}
    ).get(
        "count",
        0
    ) > 0:
        recommendations.append(
            {
                "priority": "medium",
                "area": "anomaly_review",
                "action": (
                    "Review unusual lead-value, age, and "
                    "status combinations."
                ),
                "reason": (
                    "Anomalous records may indicate data errors "
                    "or unusual sales opportunities."
                ),
            }
        )

    priority_distribution = result.get(
        "lead_priority_distribution",
        {}
    )

    if (
        priority_distribution.get(
            "critical",
            0
        ) > 0
    ):
        recommendations.append(
            {
                "priority": "critical",
                "area": "immediate_sales_action",
                "action": (
                    "Immediately assign critical-priority leads "
                    "to sales representatives."
                ),
                "reason": (
                    f"{priority_distribution.get('critical', 0)} "
                    "critical leads were identified."
                ),
            }
        )

    return recommendations


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def _build_executive_summary(
    result: Dict[str, Any]
) -> str:
    """
    Build a concise executive-level summary based only on
    observed/calculated findings.
    """
    total_leads = result.get(
        "total_leads",
        0
    )

    conversion_rate = result.get(
        "conversion_metrics",
        {}
    ).get(
        "conversion_rate_pct",
        0
    )

    pipeline_value = result.get(
        "value_analysis",
        {}
    ).get(
        "total_pipeline_value",
        0
    )

    converted = result.get(
        "converted_leads",
        0
    )

    stale = result.get(
        "lead_aging",
        {}
    ).get(
        "stale_active_leads",
        0
    )

    best_source = result.get(
        "best_source_by_conversion"
    )

    parts = [
        (
            f"The dataset contains {total_leads} leads "
            f"with {converted} converted leads."
        ),
        (
            f"Total observed pipeline value is "
            f"{pipeline_value:,.2f}, with an overall "
            f"conversion rate of {conversion_rate:.2f}%."
        ),
    ]

    if stale > 0:
        parts.append(
            f"{stale} active leads require attention because "
            f"they have reached the stale-lead threshold."
        )

    if best_source:
        parts.append(
            f"The strongest observed source by conversion "
            f"is {best_source.get('source')}."
        )

    return " ".join(parts)


# ============================================================
# MAIN AGENT
# ============================================================

def run_leads_analyzer(
    csv_content: Optional[str],
    table_json: Optional[List[Dict]],
    prompt: str
) -> Dict[str, Any]:
    """
    Analyze sales leads and lead scoring.

    Preserves the original supported inputs:

    CSV expected fields may include:
        lead_id,
        source,
        status,
        company,
        value,
        stage,
        age_days

    Or table_json with lead records.

    The prompt is preserved in the result as analysis_context.
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
                "leads-analyzer requires CSV or table_json input"
            )

        # ====================================================
        # EMPTY DATASET
        # ====================================================

        if df.empty:
            return {
                "agent": AGENT_NAME,
                "status": "success",
                "total_leads": 0,
                "status_distribution": {},
                "qualified_leads": 0,
                "converted_leads": 0,
                "value_analysis": {
                    "total_pipeline_value": 0,
                    "avg_lead_value": 0,
                    "max_lead_value": 0,
                    "median_lead_value": 0,
                },
                "by_source": {},
                "source_efficiency": {},
                "by_stage": {},
                "lead_aging": {},
                "conversion_metrics": {
                    "conversion_rate_pct": 0,
                    "lead_quality_index": 0,
                },
                "top_companies": {},
                "data_quality": {
                    "quality_status": "good",
                    "warnings": [],
                },
                "executive_summary": (
                    "The lead dataset is empty. "
                    "No lead intelligence could be calculated."
                ),
                "analysis_context": prompt,
            }

        # ====================================================
        # NORMALIZE COLUMN WHITESPACE
        # ====================================================

        df = df.copy()

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        # ====================================================
        # DISCOVER COLUMNS
        # ====================================================

        status_column = _find_status_column(df)
        source_column = _find_source_column(df)
        company_column = _find_company_column(df)
        value_column = _find_value_column(df)
        stage_column = _find_stage_column(df)
        age_column = _find_age_column(df)
        date_column = _find_date_column(df)

        # ====================================================
        # DATA QUALITY
        # ====================================================

        data_quality = _analyze_data_quality(
            df
        )

        # ====================================================
        # ORIGINAL RESULT
        # ====================================================

        result: Dict[str, Any] = {
            "agent": AGENT_NAME,
            "status": "success",

            # Preserve original behavior.
            "total_leads": int(
                len(df)
            ),

            # Advanced metadata.
            "detected_columns": {
                "status": status_column,
                "source": source_column,
                "company": company_column,
                "value": value_column,
                "stage": stage_column,
                "age_days": age_column,
                "date": date_column,
            },

            "data_quality": data_quality,
        }

        # ====================================================
        # STATUS DISTRIBUTION
        # ====================================================

        status_analysis = _analyze_statuses(
            df,
            status_column
        )

        if status_analysis:

            # Preserve original keys.
            result["status_distribution"] = (
                status_analysis[
                    "status_distribution"
                ]
            )

            result["qualified_leads"] = (
                status_analysis[
                    "qualified_leads"
                ]
            )

            result["converted_leads"] = (
                status_analysis[
                    "converted_leads"
                ]
            )

            # New advanced fields.
            result[
                "normalized_status_distribution"
            ] = status_analysis[
                "normalized_status_distribution"
            ]

            result["lost_leads"] = (
                status_analysis[
                    "lost_leads"
                ]
            )

            result["disqualified_leads"] = (
                status_analysis[
                    "disqualified_leads"
                ]
            )

            result["active_leads"] = (
                status_analysis[
                    "active_leads"
                ]
            )

        # ====================================================
        # VALUE ANALYSIS
        # ====================================================

        value_analysis = _analyze_value(
            df,
            value_column
        )

        if value_analysis:

            # Preserve original functionality.
            result["value_analysis"] = {
                "total_pipeline_value": (
                    value_analysis[
                        "total_pipeline_value"
                    ]
                ),
                "avg_lead_value": (
                    value_analysis[
                        "avg_lead_value"
                    ]
                ),
                "max_lead_value": (
                    value_analysis[
                        "max_lead_value"
                    ]
                ),
                "median_lead_value": (
                    value_analysis[
                        "median_lead_value"
                    ]
                ),
            }

            # Advanced additions.
            result[
                "value_segmentation"
            ] = {
                "value_percentiles": (
                    value_analysis[
                        "value_percentiles"
                    ]
                ),
                "high_value_leads": (
                    value_analysis[
                        "high_value_leads"
                    ]
                ),
                "very_high_value_leads": (
                    value_analysis[
                        "very_high_value_leads"
                    ]
                ),
                "high_value_pipeline": (
                    value_analysis[
                        "high_value_pipeline"
                    ]
                ),
                "high_value_pipeline_pct": (
                    value_analysis[
                        "high_value_pipeline_pct"
                    ]
                ),
                "zero_value_leads": (
                    value_analysis[
                        "zero_value_leads"
                    ]
                ),
                "positive_value_leads": (
                    value_analysis[
                        "positive_value_leads"
                    ]
                ),
            }

        # ====================================================
        # SOURCE ANALYSIS
        # ====================================================

        if source_column:

            # Preserve original source distribution.
            source_dist = (
                df[source_column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
                .value_counts()
                .to_dict()
            )

            result["by_source"] = (
                source_dist
            )

            source_analysis = _analyze_sources(
                df,
                source_column,
                status_column,
                value_column
            )

            if source_analysis:

                # Preserve original source efficiency.
                result[
                    "source_efficiency"
                ] = source_analysis.get(
                    "source_efficiency",
                    {}
                )

                # Advanced source intelligence.
                result[
                    "source_performance"
                ] = source_analysis.get(
                    "source_performance",
                    []
                )

                result[
                    "source_ranking"
                ] = source_analysis.get(
                    "source_ranking",
                    []
                )

                if (
                    "best_source_by_conversion"
                    in source_analysis
                ):
                    result[
                        "best_source_by_conversion"
                    ] = source_analysis[
                        "best_source_by_conversion"
                    ]

                if (
                    "best_source_by_pipeline_value"
                    in source_analysis
                ):
                    result[
                        "best_source_by_pipeline_value"
                    ] = source_analysis[
                        "best_source_by_pipeline_value"
                    ]

                if (
                    "best_source_by_lead_quality"
                    in source_analysis
                ):
                    result[
                        "best_source_by_lead_quality"
                    ] = source_analysis[
                        "best_source_by_lead_quality"
                    ]

        # ====================================================
        # PIPELINE STAGE ANALYSIS
        # ====================================================

        if stage_column:

            # Preserve original stage distribution.
            stage_dist = (
                df[stage_column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
                .value_counts()
                .sort_index()
                .to_dict()
            )

            result["by_stage"] = (
                stage_dist
            )

            stage_analysis = _analyze_stages(
                df,
                stage_column,
                status_column,
                value_column
            )

            result[
                "stage_performance"
            ] = stage_analysis.get(
                "stage_performance",
                []
            )

            result[
                "largest_stage_by_volume"
            ] = stage_analysis.get(
                "largest_stage_by_volume"
            )

            result[
                "highest_value_stage"
            ] = stage_analysis.get(
                "highest_value_stage"
            )

        # ====================================================
        # LEAD AGING
        # ====================================================

        if age_column:

            aging_analysis = _analyze_aging(
                df,
                age_column,
                status_column,
                value_column
            )

            # Preserve original keys.
            result["lead_aging"] = {
                "avg_age_days": (
                    aging_analysis[
                        "avg_age_days"
                    ]
                ),
                "leads_over_30_days": (
                    aging_analysis[
                        "leads_over_30_days"
                    ]
                ),
                "leads_over_60_days": (
                    aging_analysis[
                        "leads_over_60_days"
                    ]
                ),
                "leads_over_90_days": (
                    aging_analysis[
                        "leads_over_90_days"
                    ]
                ),
            }

            # Advanced aging fields.
            result[
                "lead_aging_advanced"
            ] = aging_analysis

        # ====================================================
        # CONVERSION METRICS
        # ====================================================

        if status_column:

            conversion_analysis = (
                _analyze_conversion(
                    df,
                    status_column
                )
            )

            # Preserve original keys.
            result[
                "conversion_metrics"
            ] = {
                "conversion_rate_pct": (
                    conversion_analysis.get(
                        "conversion_rate_pct",
                        0
                    )
                ),
                "lead_quality_index": (
                    conversion_analysis.get(
                        "conversion_rate_pct",
                        0
                    )
                ),
            }

            # Advanced metrics.
            result[
                "conversion_metrics"
            ].update(
                {
                    key: value
                    for key, value
                    in conversion_analysis.items()
                    if key
                    not in [
                        "conversion_rate_pct"
                    ]
                }
            )

        # ====================================================
        # COMPANY ANALYSIS
        # ====================================================

        if company_column:

            # Preserve original functionality.
            company_dist = (
                df[company_column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
                .value_counts()
                .head(DEFAULT_TOP_N)
                .to_dict()
            )

            result["top_companies"] = (
                company_dist
            )

            company_analysis = _analyze_companies(
                df,
                company_column,
                status_column,
                value_column
            )

            result[
                "company_performance"
            ] = company_analysis.get(
                "company_performance",
                []
            )

            result[
                "company_count"
            ] = company_analysis.get(
                "company_count",
                0
            )

        # ====================================================
        # FUNNEL ANALYSIS
        # ====================================================

        funnel_analysis = _analyze_funnel(
            df,
            status_column,
            stage_column
        )

        if funnel_analysis:
            result[
                "funnel_analysis"
            ] = funnel_analysis

        # ====================================================
        # LEAD SCORING
        # ====================================================

        scoring_analysis = _calculate_lead_scores(
            df,
            status_column,
            value_column,
            age_column,
            source_column
        )

        if scoring_analysis:

            result[
                "lead_scoring"
            ] = scoring_analysis.get(
                "lead_scoring",
                {}
            )

            result[
                "lead_priority_distribution"
            ] = scoring_analysis.get(
                "priority_distribution",
                {}
            )

            result[
                "average_lead_score"
            ] = scoring_analysis.get(
                "average_lead_score",
                0
            )

            result[
                "top_priority_leads"
            ] = scoring_analysis.get(
                "top_priority_leads",
                []
            )

        # ====================================================
        # PIPELINE CONCENTRATION
        # ====================================================

        concentration = _analyze_concentration(
            df,
            value_column,
            source_column
        )

        if concentration:
            result[
                "pipeline_concentration"
            ] = concentration

        # ====================================================
        # ANOMALY DETECTION
        # ====================================================

        result[
            "anomalies"
        ] = _detect_anomalies(
            df,
            value_column,
            age_column,
            status_column
        )

        # ====================================================
        # TREND ANALYSIS
        # ====================================================

        trends = _analyze_trends(
            df,
            date_column,
            status_column,
            value_column
        )

        result[
            "trend_analysis"
        ] = trends

        # ====================================================
        # FORECAST
        # ====================================================

        result[
            "forecast"
        ] = _forecast_conversions(
            trends
        )

        # ====================================================
        # DATA QUALITY CONFIDENCE
        # ====================================================

        warnings_count = len(
            data_quality.get(
                "warnings",
                []
            )
        )

        if warnings_count == 0:
            confidence = "high"

        elif warnings_count <= 2:
            confidence = "medium"

        else:
            confidence = "low"

        result[
            "confidence"
        ] = confidence

        # ====================================================
        # RISKS
        # ====================================================

        result[
            "risks"
        ] = _detect_risks(
            result
        )

        # ====================================================
        # OPPORTUNITIES
        # ====================================================

        result[
            "opportunities"
        ] = _detect_opportunities(
            result
        )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        result[
            "recommendations"
        ] = _generate_recommendations(
            result
        )

        # ====================================================
        # EXECUTIVE SUMMARY
        # ====================================================

        result[
            "executive_summary"
        ] = _build_executive_summary(
            result
        )

        # ====================================================
        # ANALYSIS CONTEXT
        # ====================================================

        # Preserve the prompt without pretending that this
        # deterministic Python layer performed LLM reasoning.
        result[
            "analysis_context"
        ] = prompt

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