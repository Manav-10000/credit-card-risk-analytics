"""
data_cleaning.py
-----------------
Cleans and validates the raw synthetic transaction data using Pandas.

Responsibilities:
    1. Load the raw CSV.
    2. Report data-quality metrics (missing values, duplicates, invalid amounts)
       BEFORE cleaning, so the dashboard can show "what we found".
    3. Produce a cleaned DataFrame that is safe to load into SQLite and to
       feed into the anomaly-detection model.

This module does not delete information silently -- it returns a
`quality_report` dict alongside the cleaned data so nothing is hidden from
the dashboard user.
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "transaction_id", "transaction_date", "customer_id", "merchant_id",
    "merchant_category", "transaction_amount", "city", "state",
    "payment_method", "card_type", "customer_age", "customer_income",
    "transaction_hour", "is_international", "is_weekend",
    "previous_transaction_amount", "distance_from_home_km",
]


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
    return df


def profile_data_quality(df: pd.DataFrame) -> dict:
    """
    Computes data-quality metrics on the RAW data, before any cleaning.
    Returned as a plain dict so it can be dropped straight into Streamlit
    metrics / tables.
    """
    total_records = len(df)

    missing_by_column = df.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].to_dict()
    total_missing_cells = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated(subset=[c for c in REQUIRED_COLUMNS if c in df.columns]).sum())

    invalid_amount_mask = df["transaction_amount"] <= 0
    invalid_amount_count = int(invalid_amount_mask.sum())

    invalid_age_mask = (df["customer_age"] < 18) | (df["customer_age"] > 100)
    invalid_age_count = int(invalid_age_mask.sum())

    missing_pct = round((total_missing_cells / (total_records * len(df.columns))) * 100, 3) if total_records else 0.0

    is_valid_overall = (
        duplicate_rows == 0
        and invalid_amount_count == 0
        and invalid_age_count == 0
        and total_missing_cells == 0
    )

    report = {
        "total_records": total_records,
        "total_columns": len(df.columns),
        "missing_by_column": missing_by_column,
        "total_missing_cells": total_missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "invalid_amount_count": invalid_amount_count,
        "invalid_age_count": invalid_age_count,
        "is_valid_overall": is_valid_overall,
    }
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a straightforward, explainable cleaning pipeline:
        - Drop exact duplicate transactions.
        - Drop transactions with non-positive amounts (data-entry errors --
          a real analyst would flag these for review rather than silently
          keep them in a spend-analysis dataset).
        - Fill missing 'city' with 'Unknown'.
        - Fill missing 'customer_income' with the median income for that
          customer's age bracket (a simple, explainable imputation).
        - Clip implausible ages to a sane range.
        - Ensure correct dtypes for downstream SQL/ML use.
    """
    cleaned = df.copy()

    # 1. Drop exact duplicates (keep first occurrence)
    subset_cols = [c for c in REQUIRED_COLUMNS if c in cleaned.columns]
    cleaned = cleaned.drop_duplicates(subset=subset_cols, keep="first")

    # 2. Remove invalid (non-positive) transaction amounts
    cleaned = cleaned[cleaned["transaction_amount"] > 0]

    # 3. Fill missing categorical values
    cleaned["city"] = cleaned["city"].fillna("Unknown")

    # 4. Impute missing income using median income per 10-year age bracket
    cleaned["age_bracket"] = (cleaned["customer_age"] // 10) * 10
    bracket_medians = cleaned.groupby("age_bracket")["customer_income"].transform("median")
    cleaned["customer_income"] = cleaned["customer_income"].fillna(bracket_medians)
    cleaned["customer_income"] = cleaned["customer_income"].fillna(cleaned["customer_income"].median())
    cleaned = cleaned.drop(columns=["age_bracket"])

    # 5. Clip implausible ages (data-entry safety net)
    cleaned["customer_age"] = cleaned["customer_age"].clip(lower=18, upper=100)

    # 6. Enforce dtypes
    cleaned["transaction_amount"] = cleaned["transaction_amount"].astype(float)
    cleaned["previous_transaction_amount"] = cleaned["previous_transaction_amount"].astype(float)
    cleaned["distance_from_home_km"] = cleaned["distance_from_home_km"].astype(float)
    cleaned["customer_income"] = cleaned["customer_income"].astype(float)
    cleaned["is_international"] = cleaned["is_international"].astype(int)
    cleaned["is_weekend"] = cleaned["is_weekend"].astype(int)
    cleaned["transaction_hour"] = cleaned["transaction_hour"].astype(int)

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def run_cleaning_pipeline(csv_path: str):
    """Convenience wrapper: load -> profile -> clean. Returns (cleaned_df, quality_report)."""
    raw_df = load_raw_data(csv_path)
    quality_report = profile_data_quality(raw_df)
    cleaned_df = clean_data(raw_df)
    quality_report["total_records_after_cleaning"] = len(cleaned_df)
    quality_report["records_removed"] = quality_report["total_records"] - len(cleaned_df)
    return cleaned_df, quality_report


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "transactions.csv")
    cleaned_df, report = run_cleaning_pipeline(csv_path)
    print("Data Quality Report (on raw data):")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nCleaned dataset shape: {cleaned_df.shape}")
