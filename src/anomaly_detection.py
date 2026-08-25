"""
anomaly_detection.py
---------------------
Uses scikit-learn's IsolationForest to flag POTENTIALLY UNUSUAL transactions.

IMPORTANT (read this before using the output anywhere):
    - This is UNSUPERVISED anomaly detection. The model is never told which
      transactions are "fraud" -- it only learns what "typical" transactions
      look like and flags the ones that are statistically different.
    - A flagged transaction is NOT confirmed fraud. It simply means the
      transaction's amount, timing, distance-from-home, or other numeric
      pattern differs from the bulk of the data. Plenty of legitimate
      transactions (e.g. a big one-off purchase, a trip abroad) can be
      flagged too. In a real risk team, these flags are a starting point
      for investigation, not a verdict.

Features used (kept simple and explainable on purpose):
    - transaction_amount
    - transaction_hour
    - distance_from_home_km
    - previous_transaction_amount
    - customer_income
    - is_international
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "transaction_amount",
    "transaction_hour",
    "distance_from_home_km",
    "previous_transaction_amount",
    "customer_income",
    "is_international",
]

# Expected anomaly proportion. This is a modeling assumption, not a claim
# about real-world fraud rates -- it just tells Isolation Forest roughly
# how many points to treat as outliers.
CONTAMINATION = 0.02
RANDOM_SEED = 42


def detect_anomalies(df: pd.DataFrame, contamination: float = CONTAMINATION) -> pd.DataFrame:
    """
    Fits an IsolationForest on the given DataFrame and returns a COPY of the
    DataFrame with two new columns:
        - anomaly_score: raw model score (lower = more unusual)
        - is_anomaly: 1 if flagged as a potentially unusual transaction, else 0

    The model is fit fresh each time this function is called (no persisted
    model file), which keeps the project simple and avoids stale-model bugs
    for a portfolio project. For a production system you would train once,
    save the model (e.g. with joblib), and re-use it for scoring new data.
    """
    result = df.copy()

    features = result[FEATURE_COLUMNS].copy()
    features = features.fillna(features.median(numeric_only=True))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(scaled_features)

    # decision_function: higher = more "normal", lower/negative = more anomalous
    result["anomaly_score"] = model.decision_function(scaled_features)
    # predict: -1 = anomaly, 1 = normal -> convert to a friendlier 1/0 flag
    raw_predictions = model.predict(scaled_features)
    result["is_anomaly"] = (raw_predictions == -1).astype(int)

    return result


def summarize_anomalies(scored_df: pd.DataFrame) -> dict:
    """Small helper to compute the anomaly KPIs shown on the dashboard."""
    total = len(scored_df)
    anomaly_count = int(scored_df["is_anomaly"].sum())
    anomaly_rate = round((anomaly_count / total) * 100, 3) if total else 0.0
    return {
        "total_transactions": total,
        "anomaly_count": anomaly_count,
        "anomaly_rate_pct": anomaly_rate,
        "avg_anomaly_score": round(scored_df["anomaly_score"].mean(), 4),
        "avg_amount_normal": round(scored_df.loc[scored_df["is_anomaly"] == 0, "transaction_amount"].mean(), 2),
        "avg_amount_anomaly": round(scored_df.loc[scored_df["is_anomaly"] == 1, "transaction_amount"].mean(), 2)
        if anomaly_count else 0.0,
    }


if __name__ == "__main__":
    import os
    from data_cleaning import run_cleaning_pipeline

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "transactions.csv")

    cleaned_df, _ = run_cleaning_pipeline(csv_path)
    scored_df = detect_anomalies(cleaned_df)
    summary = summarize_anomalies(scored_df)

    print("Anomaly Detection Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Sanity check: how many of our intentionally-injected 'unusual' rows
    # did the unsupervised model actually catch? (For our own curiosity only
    # -- the model never sees ground_truth_unusual during training.)
    if "ground_truth_unusual" in scored_df.columns:
        caught = scored_df[(scored_df["ground_truth_unusual"] == 1) & (scored_df["is_anomaly"] == 1)]
        total_injected = scored_df["ground_truth_unusual"].sum()
        print(f"\n  (Dev check) Injected unusual transactions caught by model: "
              f"{len(caught)} / {total_injected}")
