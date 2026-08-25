"""
analysis.py
-----------
Loads the cleaned (and anomaly-scored) transaction data into a local SQLite
database, and provides SQL-based analysis functions used by the dashboard.

Why SQLite here?
    It requires no server setup (great for a portfolio project that anyone
    can clone and run), while still letting us demonstrate real SQL:
    GROUP BY, aggregate functions, window-style ranking via ORDER BY/LIMIT,
    and CASE WHEN logic -- the same SQL skills used against Snowflake/
    Postgres/Redshift in a production analytics role.
"""

import os
import sqlite3
import pandas as pd

TABLE_NAME = "transactions"


def get_connection(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def load_dataframe_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = TABLE_NAME):
    """Writes the DataFrame to SQLite, replacing any existing table."""
    conn = get_connection(db_path)
    try:
        df_to_store = df.copy()
        # SQLite has no native datetime type -- store as ISO string
        df_to_store["transaction_date"] = df_to_store["transaction_date"].astype(str)
        df_to_store.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def run_query(query: str, db_path: str) -> pd.DataFrame:
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Named SQL queries used throughout the dashboard. Kept here as Python
# strings (mirrored in sql/analysis.sql for a plain-SQL reference file that
# reviewers/interviewers can read without running any Python).
# ---------------------------------------------------------------------------

QUERIES = {

    "monthly_transaction_value": f"""
        SELECT
            strftime('%Y-%m', transaction_date) AS month,
            COUNT(*) AS transaction_count,
            ROUND(SUM(transaction_amount), 2) AS total_value,
            ROUND(AVG(transaction_amount), 2) AS avg_value
        FROM {TABLE_NAME}
        GROUP BY month
        ORDER BY month;
    """,

    "transaction_count_by_category": f"""
        SELECT
            merchant_category,
            COUNT(*) AS transaction_count
        FROM {TABLE_NAME}
        GROUP BY merchant_category
        ORDER BY transaction_count DESC;
    """,

    "spend_by_category": f"""
        SELECT
            merchant_category,
            ROUND(SUM(transaction_amount), 2) AS total_spend,
            ROUND(AVG(transaction_amount), 2) AS avg_spend
        FROM {TABLE_NAME}
        GROUP BY merchant_category
        ORDER BY total_spend DESC;
    """,

    "top_customers_by_value": f"""
        SELECT
            customer_id,
            COUNT(*) AS transaction_count,
            ROUND(SUM(transaction_amount), 2) AS total_spend
        FROM {TABLE_NAME}
        GROUP BY customer_id
        ORDER BY total_spend DESC
        LIMIT 10;
    """,

    "top_merchants_by_value": f"""
        SELECT
            merchant_id,
            merchant_category,
            COUNT(*) AS transaction_count,
            ROUND(SUM(transaction_amount), 2) AS total_value
        FROM {TABLE_NAME}
        GROUP BY merchant_id, merchant_category
        ORDER BY total_value DESC
        LIMIT 10;
    """,

    "weekend_vs_weekday": f"""
        SELECT
            CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            COUNT(*) AS transaction_count,
            ROUND(SUM(transaction_amount), 2) AS total_value,
            ROUND(AVG(transaction_amount), 2) AS avg_value
        FROM {TABLE_NAME}
        GROUP BY day_type;
    """,

    "international_vs_domestic": f"""
        SELECT
            CASE WHEN is_international = 1 THEN 'International' ELSE 'Domestic' END AS txn_type,
            COUNT(*) AS transaction_count,
            ROUND(SUM(transaction_amount), 2) AS total_value,
            ROUND(AVG(transaction_amount), 2) AS avg_value
        FROM {TABLE_NAME}
        GROUP BY txn_type;
    """,

    "high_value_transactions": f"""
        SELECT
            transaction_id, transaction_date, customer_id, merchant_id,
            merchant_category, transaction_amount, state, is_international
        FROM {TABLE_NAME}
        WHERE transaction_amount >= 1000
        ORDER BY transaction_amount DESC
        LIMIT 100;
    """,

    "value_by_state": f"""
        SELECT
            state,
            COUNT(*) AS transaction_count,
            ROUND(SUM(transaction_amount), 2) AS total_value,
            ROUND(AVG(transaction_amount), 2) AS avg_value
        FROM {TABLE_NAME}
        GROUP BY state
        ORDER BY total_value DESC;
    """,
}


def run_all_named_queries(db_path: str) -> dict:
    """Runs every query in QUERIES and returns {name: DataFrame}."""
    return {name: run_query(sql, db_path) for name, sql in QUERIES.items()}


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base_dir, "src"))
    from data_cleaning import run_cleaning_pipeline
    from anomaly_detection import detect_anomalies

    csv_path = os.path.join(base_dir, "data", "transactions.csv")
    db_path = os.path.join(base_dir, "database", "transactions.db")

    cleaned_df, _ = run_cleaning_pipeline(csv_path)
    scored_df = detect_anomalies(cleaned_df)

    load_dataframe_to_sqlite(scored_df, db_path)
    print(f"Loaded {len(scored_df):,} rows into {db_path}")

    results = run_all_named_queries(db_path)
    for name, result_df in results.items():
        print(f"\n--- {name} ---")
        print(result_df.head(5).to_string(index=False))
