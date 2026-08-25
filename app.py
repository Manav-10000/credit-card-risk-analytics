"""
app.py
------
Credit Card Transaction Risk & Analytics Dashboard.

A Streamlit application that:
    1. Loads and cleans a synthetic transaction dataset (Pandas).
    2. Persists it into SQLite and runs SQL-based analysis.
    3. Scores every transaction with an IsolationForest model to flag
       potentially unusual transactions (Scikit-learn).
    4. Presents everything as an interactive dashboard (Plotly + Streamlit).

Run with:
    streamlit run app.py

NOTE ON DATA: All data in this app is SYNTHETIC and randomly generated
(see src/data_generator.py). It does not represent real Amex, cardholder,
or merchant data. Transactions flagged as "anomalies" are statistical
outliers only -- they are NOT confirmed fraud.
"""

import os
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_generator import generate_transactions
from data_cleaning import profile_data_quality, clean_data
from anomaly_detection import detect_anomalies, summarize_anomalies, FEATURE_COLUMNS
from analysis import load_dataframe_to_sqlite, run_all_named_queries

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
DB_PATH = os.path.join(BASE_DIR, "database", "transactions.db")

st.set_page_config(
    page_title="Credit Card Transaction Risk & Analytics Dashboard",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached so the app doesn't redo work on every interaction)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_or_generate_raw_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["transaction_date"])
    df = generate_transactions()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


@st.cache_data(show_spinner=False)
def prepare_data(raw_df: pd.DataFrame):
    quality_report = profile_data_quality(raw_df)
    cleaned_df = clean_data(raw_df)
    quality_report["total_records_after_cleaning"] = len(cleaned_df)
    quality_report["records_removed"] = quality_report["total_records"] - len(cleaned_df)

    scored_df = detect_anomalies(cleaned_df)
    return scored_df, quality_report


@st.cache_data(show_spinner=False)
def persist_and_query(scored_df: pd.DataFrame):
    load_dataframe_to_sqlite(scored_df, DB_PATH)
    sql_results = run_all_named_queries(DB_PATH)
    return sql_results


raw_df = load_or_generate_raw_data(DATA_PATH)
scored_df, quality_report = prepare_data(raw_df)
sql_results = persist_and_query(scored_df)
scored_df["transaction_date"] = pd.to_datetime(scored_df["transaction_date"])

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 Filters")

min_date = scored_df["transaction_date"].min().date()
max_date = scored_df["transaction_date"].max().date()
date_range = st.sidebar.date_input(
    "Transaction date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

category_options = sorted(scored_df["merchant_category"].unique())
selected_categories = st.sidebar.multiselect(
    "Merchant category", category_options, default=category_options
)

state_options = sorted(scored_df["state"].unique())
selected_states = st.sidebar.multiselect(
    "State", state_options, default=state_options
)

card_type_options = sorted(scored_df["card_type"].unique())
selected_card_types = st.sidebar.multiselect(
    "Card type", card_type_options, default=card_type_options
)

payment_options = sorted(scored_df["payment_method"].unique())
selected_payments = st.sidebar.multiselect(
    "Payment method", payment_options, default=payment_options
)

intl_choice = st.sidebar.radio(
    "International / Domestic", ["All", "International only", "Domestic only"], index=0
)

anomaly_choice = st.sidebar.radio(
    "Anomaly status", ["All", "Flagged only", "Not flagged only"], index=0
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ All data is synthetic. Flagged transactions are statistical "
    "outliers, **not confirmed fraud**."
)

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = scored_df[
    (scored_df["transaction_date"].dt.date >= start_date)
    & (scored_df["transaction_date"].dt.date <= end_date)
    & (scored_df["merchant_category"].isin(selected_categories))
    & (scored_df["state"].isin(selected_states))
    & (scored_df["card_type"].isin(selected_card_types))
    & (scored_df["payment_method"].isin(selected_payments))
]

if intl_choice == "International only":
    filtered_df = filtered_df[filtered_df["is_international"] == 1]
elif intl_choice == "Domestic only":
    filtered_df = filtered_df[filtered_df["is_international"] == 0]

if anomaly_choice == "Flagged only":
    filtered_df = filtered_df[filtered_df["is_anomaly"] == 1]
elif anomaly_choice == "Not flagged only":
    filtered_df = filtered_df[filtered_df["is_anomaly"] == 0]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💳 Credit Card Transaction Risk & Analytics Dashboard")
st.caption(
    "Synthetic transaction data · Pandas + SQLite/SQL + Isolation Forest + Streamlit + Plotly"
)

if filtered_df.empty:
    st.warning("No transactions match the current filters. Try widening your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

kpi_summary = summarize_anomalies(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{kpi_summary['total_transactions']:,}")
col2.metric("Total Transaction Value", f"${filtered_df['transaction_amount'].sum():,.2f}")
col3.metric("Average Transaction", f"${filtered_df['transaction_amount'].mean():,.2f}")
col4.metric(
    "Anomaly Rate",
    f"{kpi_summary['anomaly_rate_pct']}%",
    help="Share of transactions flagged as potentially unusual by the Isolation Forest model.",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs: Overview / Anomalies / Data Quality / SQL Insights
# ---------------------------------------------------------------------------

tab_overview, tab_anomalies, tab_quality, tab_sql = st.tabs(
    ["📊 Overview", "🚩 Anomaly Detection", "🧹 Data Quality", "🗄️ SQL Insights"]
)

# ---------------- Overview tab ----------------
with tab_overview:
    st.subheader("Monthly Transaction Trend")
    monthly = (
        filtered_df.set_index("transaction_date")
        .resample("MS")["transaction_amount"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "transaction_count", "sum": "total_value"})
    )
    fig_monthly = px.line(
        monthly, x="transaction_date", y="total_value", markers=True,
        title="Total Transaction Value by Month",
        labels={"transaction_date": "Month", "total_value": "Total Value ($)"},
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Spending by Merchant Category")
        cat_spend = (
            filtered_df.groupby("merchant_category")["transaction_amount"]
            .sum().sort_values(ascending=False).reset_index()
        )
        fig_cat_spend = px.bar(
            cat_spend, x="merchant_category", y="transaction_amount",
            title="Total Spend by Category",
            labels={"merchant_category": "Category", "transaction_amount": "Total Spend ($)"},
        )
        st.plotly_chart(fig_cat_spend, use_container_width=True)

    with col_b:
        st.subheader("Transaction Count by Category")
        cat_count = (
            filtered_df.groupby("merchant_category").size()
            .sort_values(ascending=False).reset_index(name="transaction_count")
        )
        fig_cat_count = px.bar(
            cat_count, x="merchant_category", y="transaction_count",
            title="Number of Transactions by Category",
            labels={"merchant_category": "Category", "transaction_count": "Transaction Count"},
        )
        st.plotly_chart(fig_cat_count, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Transactions by State")
        state_count = (
            filtered_df.groupby("state").size()
            .sort_values(ascending=False).reset_index(name="transaction_count")
        )
        fig_state_count = px.bar(
            state_count, x="state", y="transaction_count",
            title="Transaction Count by State",
        )
        st.plotly_chart(fig_state_count, use_container_width=True)

    with col_d:
        st.subheader("Transaction Value by State")
        state_value = (
            filtered_df.groupby("state")["transaction_amount"].sum()
            .sort_values(ascending=False).reset_index()
        )
        fig_state_value = px.bar(
            state_value, x="state", y="transaction_amount",
            title="Total Value by State",
            labels={"transaction_amount": "Total Value ($)"},
        )
        st.plotly_chart(fig_state_value, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("International vs Domestic")
        intl_summary = (
            filtered_df.assign(
                txn_type=np.where(filtered_df["is_international"] == 1, "International", "Domestic")
            )
            .groupby("txn_type")["transaction_amount"]
            .agg(["count", "sum"]).reset_index()
            .rename(columns={"count": "transaction_count", "sum": "total_value"})
        )
        fig_intl = px.pie(
            intl_summary, names="txn_type", values="transaction_count",
            title="Share of Transactions: International vs Domestic",
        )
        st.plotly_chart(fig_intl, use_container_width=True)

    with col_f:
        st.subheader("High-Value Transactions (≥ $1,000)")
        high_value = filtered_df[filtered_df["transaction_amount"] >= 1000]
        fig_high_value = px.histogram(
            high_value, x="transaction_amount", nbins=30,
            title=f"Distribution of High-Value Transactions (n={len(high_value):,})",
            labels={"transaction_amount": "Transaction Amount ($)"},
        )
        st.plotly_chart(fig_high_value, use_container_width=True)

# ---------------- Anomaly Detection tab ----------------
with tab_anomalies:
    st.subheader("Potentially Unusual Transactions")
    st.info(
        "These transactions were flagged by an **unsupervised Isolation Forest model** "
        "because their pattern (amount, timing, distance from home, etc.) differs from "
        "typical transactions. This is **not** a fraud determination — it is a "
        "starting point for further review.",
        icon="ℹ️",
    )

    anomaly_summary = summarize_anomalies(filtered_df)
    a1, a2, a3 = st.columns(3)
    a1.metric("Flagged Transactions", f"{anomaly_summary['anomaly_count']:,}")
    a2.metric("Anomaly Rate", f"{anomaly_summary['anomaly_rate_pct']}%")
    a3.metric(
        "Avg Amount: Flagged vs Normal",
        f"${anomaly_summary['avg_amount_anomaly']:,.2f}",
        delta=f"vs ${anomaly_summary['avg_amount_normal']:,.2f} normal",
        delta_color="off",
    )

    st.subheader("Anomaly Score Distribution")
    st.caption("Lower scores indicate transactions the model considers more unusual.")
    fig_score_dist = px.histogram(
        filtered_df, x="anomaly_score", color=filtered_df["is_anomaly"].map({0: "Normal", 1: "Flagged"}),
        nbins=50, title="Isolation Forest Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly Score", "color": "Status"},
        color_discrete_map={"Normal": "#4C78A8", "Flagged": "#E45756"},
    )
    st.plotly_chart(fig_score_dist, use_container_width=True)

    st.subheader("Flagged Transactions Table")
    flagged_df = filtered_df[filtered_df["is_anomaly"] == 1].sort_values("anomaly_score")
    display_cols = [
        "transaction_id", "transaction_date", "customer_id", "merchant_id",
        "merchant_category", "transaction_amount", "transaction_hour",
        "distance_from_home_km", "is_international", "anomaly_score",
    ]
    st.dataframe(
        flagged_df[display_cols].reset_index(drop=True),
        use_container_width=True, height=350,
    )

    with st.expander("Which features does the model use, and why?"):
        st.markdown(
            """
            The Isolation Forest looks at each transaction's:
            - **transaction_amount** — unusually large or small charges stand out.
            - **transaction_hour** — purchases in the middle of the night are rarer.
            - **distance_from_home_km** — transactions far from a customer's usual
              area are less common.
            - **previous_transaction_amount** — a sudden jump from a customer's
              recent spending pattern is a signal.
            - **customer_income** — helps the model learn what a "typical" spend
              level looks like for different income bands.
            - **is_international** — cross-border transactions are statistically rarer.

            The model doesn't use rules like "amount > $5,000 = flag". Instead, it
            builds random decision trees that isolate points, and points that get
            isolated in fewer splits (i.e., they're "easy to separate" from the rest
            of the data) get a lower, more anomalous score.
            """
        )

# ---------------- Data Quality tab ----------------
with tab_quality:
    st.subheader("Data Quality Report (Raw Data, Before Cleaning)")
    st.caption(
        "This reflects the full raw dataset (not affected by sidebar filters), "
        "so you can see exactly what the cleaning pipeline found and fixed."
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Raw Records", f"{quality_report['total_records']:,}")
    q2.metric("Missing Values (cells)", f"{quality_report['total_missing_cells']:,}")
    q3.metric("Duplicate Rows", f"{quality_report['duplicate_rows']:,}")
    q4.metric("Invalid Amounts", f"{quality_report['invalid_amount_count']:,}")

    status_ok = quality_report["is_valid_overall"]
    if status_ok:
        st.success("✅ Validation status: No data quality issues detected.")
    else:
        st.warning(
            "⚠️ Validation status: Issues detected in raw data — "
            "resolved automatically by the cleaning pipeline (see below)."
        )

    st.markdown("#### Missing Values by Column")
    if quality_report["missing_by_column"]:
        missing_df = pd.DataFrame(
            list(quality_report["missing_by_column"].items()),
            columns=["column", "missing_count"],
        )
        fig_missing = px.bar(
            missing_df, x="column", y="missing_count",
            title="Missing Values by Column",
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.write("No missing values found.")

    st.markdown("#### Cleaning Actions Taken")
    st.markdown(
        f"""
        - **Duplicate rows removed:** {quality_report['duplicate_rows']:,}
        - **Invalid (≤ 0) transaction amounts removed:** {quality_report['invalid_amount_count']:,}
        - **Missing `city` values:** filled with `"Unknown"`
        - **Missing `customer_income` values:** filled with the median income
          for that customer's age bracket
        - **Records before cleaning:** {quality_report['total_records']:,}
        - **Records after cleaning:** {quality_report['total_records_after_cleaning']:,}
        - **Total records removed:** {quality_report['records_removed']:,}
        """
    )

# ---------------- SQL Insights tab ----------------
with tab_sql:
    st.subheader("SQL-Based Analysis")
    st.caption(
        "These results come directly from SQL queries run against the SQLite "
        "database (see sql/analysis.sql). They reflect the full cleaned + "
        "scored dataset, not the sidebar filters above."
    )

    query_display_names = {
        "monthly_transaction_value": "Monthly Transaction Value",
        "transaction_count_by_category": "Transaction Count by Category",
        "spend_by_category": "Average Transaction Value by Category",
        "top_customers_by_value": "Top 10 Customers by Transaction Value",
        "top_merchants_by_value": "Top 10 Merchants by Transaction Value",
        "weekend_vs_weekday": "Weekend vs Weekday Transactions",
        "international_vs_domestic": "International vs Domestic Transactions",
        "high_value_transactions": "High-Value Transactions (≥ $1,000)",
        "value_by_state": "Transaction Value by State",
    }

    selected_query = st.selectbox(
        "Choose a SQL query to view", list(query_display_names.values())
    )
    reverse_lookup = {v: k for k, v in query_display_names.items()}
    query_key = reverse_lookup[selected_query]

    st.dataframe(sql_results[query_key], use_container_width=True, height=400)

    with st.expander("View the underlying SQL"):
        from analysis import QUERIES
        st.code(QUERIES[query_key].strip(), language="sql")

st.markdown("---")
st.caption(
    "Built as a portfolio project. Dataset is synthetic; anomalies are "
    "unsupervised model flags, not confirmed fraud."
)
