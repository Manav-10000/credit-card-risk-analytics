# 💳 Credit Card Transaction Risk & Analytics Dashboard

A portfolio project that simulates a real-world **Risk & Analytics** workflow at a
payments company: cleaning transaction data, analyzing it with SQL, flagging
statistically unusual transactions with machine learning, and presenting it
all in an interactive dashboard.

> ⚠️ **All data in this project is synthetic and randomly generated.**
> No real cardholder, merchant, or transaction data is used anywhere.
> Transactions flagged by the model are **potentially unusual / statistical
> outliers — not confirmed fraud.**

---

## 1. Project Overview

This project builds an end-to-end analytics pipeline over ~20,000 synthetic
credit-card transactions:

1. **Generate** a realistic synthetic transaction dataset.
2. **Clean & validate** it with Pandas (missing values, duplicates, invalid amounts).
3. **Store & query** it in SQLite using SQL (trends, category/state breakdowns, top customers/merchants).
4. **Score** every transaction with an Isolation Forest model to flag potentially unusual activity.
5. **Visualize** everything in an interactive Streamlit + Plotly dashboard with sidebar filters.

## 2. Business Problem

Payments and card companies (like American Express) process millions of
transactions daily and need to:

- Understand **spending trends** across time, geography, and merchant category.
- Maintain **high data quality** before any downstream reporting or modeling.
- **Surface unusual transaction patterns early** so risk analysts can review
  them, without needing labeled fraud data (which is rare, delayed, and
  sensitive).

This project simulates that workflow end-to-end on synthetic data, from raw
transactions to an analyst-facing dashboard.

## 3. Objectives

- Build a **realistic, messy** synthetic dataset (with real-world issues:
  missing values, duplicates, invalid amounts) and clean it properly.
- Demonstrate **SQL analytical skills** (aggregation, grouping, ranking, `CASE WHEN`).
- Apply **unsupervised anomaly detection** (Isolation Forest) in an explainable way.
- Deliver insights through a **decision-maker-friendly dashboard**, not just notebooks.

## 4. Architecture

```
                ┌─────────────────────┐
                │ data_generator.py    │  Synthetic data (~20k transactions)
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ data_cleaning.py     │  Pandas: validate, dedupe, impute
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ anomaly_detection.py │  Isolation Forest → anomaly_score, is_anomaly
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ analysis.py          │  Load into SQLite, run SQL queries
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ app.py (Streamlit)   │  KPI cards, charts, anomaly & data-quality views
                └─────────────────────┘
```

## 5. Technologies

| Layer               | Tool                                   |
|---------------------|-----------------------------------------|
| Data generation/cleaning | Python, Pandas, NumPy               |
| Storage & analysis  | SQLite, SQL                             |
| Anomaly detection    | Scikit-learn (`IsolationForest`)       |
| Dashboard            | Streamlit                              |
| Visualizations       | Plotly                                 |

## 6. Project Structure

```
credit-card-risk-analytics/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
├── data/
│   └── transactions.csv       # Generated synthetic dataset
├── database/
│   └── transactions.db        # SQLite database (generated on run)
├── src/
│   ├── data_generator.py      # Creates the synthetic dataset
│   ├── data_cleaning.py       # Cleaning + data quality profiling
│   ├── analysis.py            # SQLite loading + SQL queries
│   └── anomaly_detection.py   # Isolation Forest scoring
└── sql/
    └── analysis.sql           # Plain-SQL reference (same queries as analysis.py)
```

## 7. How Anomaly Detection Works

The dashboard uses **Isolation Forest**, an unsupervised algorithm that
doesn't need any labeled "fraud"/"not fraud" examples. Instead, it:

1. Builds many random decision trees that repeatedly split the data on
   random features/thresholds.
2. Points that are **isolated in fewer splits** (i.e., easy to separate from
   the rest of the data) are considered more anomalous.
3. Each transaction gets an **anomaly_score** (lower = more unusual) and an
   **is_anomaly** flag (1 = flagged).

**Features used** (kept simple and explainable on purpose):
`transaction_amount`, `transaction_hour`, `distance_from_home_km`,
`previous_transaction_amount`, `customer_income`, `is_international`.

The model is configured with a `contamination` rate of 2%, meaning it
expects roughly 2% of transactions to be outliers — this is a modeling
assumption for this synthetic dataset, not a real-world fraud rate.

**Why this matters for the business problem:** flagging is a *first pass*.
A real risk team would route flagged transactions to further review (device
data, customer contact history, merchant risk score, etc.) rather than
act on the model output alone.

## 8. How to Run

### Prerequisites
- Python 3.9+

### Setup

```bash
# 1. Move into the project folder
cd credit-card-risk-analytics

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) regenerate the synthetic dataset from scratch
python src/data_generator.py

# 5. (Optional) run the cleaning + SQL + anomaly pipeline standalone
python src/data_cleaning.py
python src/anomaly_detection.py
python src/analysis.py

# 6. Launch the dashboard
streamlit run app.py
```

The dashboard will open automatically in your browser (usually at
`http://localhost:8501`). If `data/transactions.csv` doesn't exist yet,
`app.py` will generate it automatically the first time it runs.

## 9. Limitations

- **Synthetic data**: patterns are randomly generated and simplified; real
  transaction data has far more complexity (merchant risk history, device
  fingerprints, chargebacks, etc.).
- **No labeled ground truth**: because this is unsupervised, we can't report
  precision/recall against "real" fraud — only that the model isolates
  transactions that are statistically different from the norm.
- **Model is retrained on every run**: for simplicity, there's no persisted/
  versioned model. A production system would train once, validate, and
  serve a saved model.
- **Single-machine SQLite**: fine for a portfolio project; a production
  system would use a data warehouse (Snowflake, Redshift, BigQuery, etc.).

## 10. 60-Second Interview Explanation

> "I built an end-to-end analytics dashboard that simulates how a risk team
> might monitor credit-card transactions. I started by generating a
> realistic synthetic dataset of about 20,000 transactions, intentionally
> including messy data — missing values, duplicates, invalid amounts — so I
> could build a proper Pandas cleaning pipeline that profiles and fixes
> those issues. I loaded the cleaned data into SQLite and wrote SQL queries
> to analyze spending trends by month, category, state, and customer. Then
> I used an Isolation Forest model — an unsupervised anomaly-detection
> algorithm — to flag transactions that look statistically unusual based on
> amount, timing, distance from home, and a few other features. Importantly,
> I'm careful to call these 'potentially unusual transactions,' not
> confirmed fraud, since the model has no labeled fraud data to learn from.
> Finally, I wrapped all of this in a Streamlit dashboard with KPI cards,
> Plotly charts, and filters, so a non-technical stakeholder could explore
> the data and drill into flagged transactions themselves."

## 11. Likely Interview Questions & Answers

**Q1: Why did you choose Isolation Forest instead of a supervised model?**
A: Real-world fraud datasets are heavily imbalanced and rarely have
reliable, complete labels. Isolation Forest doesn't need labeled data — it
learns what "normal" looks like and flags points that are easy to isolate
from the rest. It's a practical first step when you don't have (or can't
fully trust) fraud labels, which is common in early-stage risk monitoring.

**Q2: How do you know the model is actually working, since you don't have real fraud labels?**
A: In this project, I intentionally injected a small set of "unusual"
transactions during data generation (very large amounts, odd hours, far
from home) to sanity-check the model. It successfully flagged the large
majority of those. In production, you'd validate similarly using historical
confirmed-fraud cases, analyst feedback loops, and back-testing.

**Q3: Why SQLite instead of a "real" database?**
A: SQLite requires zero setup, which makes the project easy to run and
review for anyone cloning the repo. The SQL itself — `GROUP BY`, aggregates,
`CASE WHEN`, ranking with `ORDER BY`/`LIMIT` — is the same SQL you'd write
against Postgres, Snowflake, or Redshift; only the connection details change.

**Q4: How would you scale this to real, larger-scale data?**
A: I'd move storage to a proper data warehouse, persist and version the
trained model (e.g., with `joblib` or MLflow) rather than retraining every
run, add a scheduled batch or streaming scoring pipeline, and add
monitoring for model drift as transaction patterns change over time.

**Q5: What features did you choose for the anomaly model, and why?**
A: I kept it intentionally simple and explainable: transaction amount,
hour of day, distance from home, previous transaction amount, customer
income, and international flag. These are all things a risk analyst
intuitively understands as "signals" — a large late-night purchase far from
home is a very different pattern than a routine grocery run. I standardized
the features before fitting the model so no single feature (like income,
which has a much larger scale) dominates the distance calculations
Isolation Forest relies on internally.

---

## GitHub-Style Short README (for repo top)

> **Credit Card Transaction Risk & Analytics Dashboard** — an end-to-end
> Python project that generates a synthetic 20K-row credit-card transaction
> dataset, cleans and validates it with Pandas, analyzes it with SQL
> (SQLite), flags potentially unusual transactions using an unsupervised
> Isolation Forest model, and presents everything in an interactive
> Streamlit + Plotly dashboard with KPI cards, trend charts, and drill-down
> filters. Built as a Data Analyst / Risk & Analytics portfolio project.
> **Note:** all data is synthetic; flagged transactions are statistical
> outliers, not confirmed fraud.

**Tech stack:** Python · Pandas · NumPy · SQLite/SQL · Scikit-learn ·
Streamlit · Plotly
