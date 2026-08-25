"""
data_generator.py
------------------
Generates a synthetic credit-card transaction dataset for the
Credit Card Transaction Risk & Analytics Dashboard project.

IMPORTANT: This data is 100% synthetic (randomly generated). It does not
represent any real customers, merchants, or transactions. It is built
purely so the rest of the pipeline (cleaning, SQL analysis, anomaly
detection, dashboard) has realistic-looking data to work with.

Run directly to regenerate data/transactions.csv:
    python src/data_generator.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Fixed seed -> reproducible dataset every time this script is run
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_TRANSACTIONS = 20000
N_CUSTOMERS = 1500
N_MERCHANTS = 400

MERCHANT_CATEGORIES = [
    "Grocery", "Restaurants", "Travel", "Electronics", "Fashion",
    "Fuel", "Entertainment", "Healthcare", "Utilities", "Online Retail",
    "Home & Furniture", "Education",
]

# Typical average spend per category (used as a base to sample amounts from,
# so the data "feels" realistic rather than pure noise)
CATEGORY_AVG_AMOUNT = {
    "Grocery": 55, "Restaurants": 40, "Travel": 320, "Electronics": 260,
    "Fashion": 90, "Fuel": 45, "Entertainment": 35, "Healthcare": 150,
    "Utilities": 110, "Online Retail": 70, "Home & Furniture": 200,
    "Education": 180,
}

US_STATES_CITIES = {
    "NY": "New York", "CA": "Los Angeles", "IL": "Chicago", "TX": "Houston",
    "AZ": "Phoenix", "PA": "Philadelphia", "TX2": "San Antonio", "CA2": "San Diego",
    "TX3": "Dallas", "CA3": "San Jose", "FL": "Miami", "WA": "Seattle",
    "CO": "Denver", "MA": "Boston", "GA": "Atlanta", "OR": "Portland",
    "NV": "Las Vegas", "MI": "Detroit", "NC": "Charlotte", "OH": "Columbus",
}
# Clean up duplicate-looking keys (used above only to get city variety)
STATE_CITY_PAIRS = [
    ("NY", "New York"), ("CA", "Los Angeles"), ("IL", "Chicago"), ("TX", "Houston"),
    ("AZ", "Phoenix"), ("PA", "Philadelphia"), ("TX", "San Antonio"), ("CA", "San Diego"),
    ("TX", "Dallas"), ("CA", "San Jose"), ("FL", "Miami"), ("WA", "Seattle"),
    ("CO", "Denver"), ("MA", "Boston"), ("GA", "Atlanta"), ("OR", "Portland"),
    ("NV", "Las Vegas"), ("MI", "Detroit"), ("NC", "Charlotte"), ("OH", "Columbus"),
]

PAYMENT_METHODS = ["Chip", "Swipe", "Contactless (Tap)", "Online"]
CARD_TYPES = ["Platinum", "Gold", "Green", "Business Green", "Business Platinum"]

START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2026, 8, 23)


def _random_dates(n):
    total_seconds = int((END_DATE - START_DATE).total_seconds())
    offsets = np.random.randint(0, total_seconds, size=n)
    return [START_DATE + timedelta(seconds=int(s)) for s in offsets]


def generate_customers(n_customers=N_CUSTOMERS):
    """Pre-generate a pool of customers with stable age/income, so the same
    customer_id has consistent demographic info across all their transactions."""
    customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, n_customers + 1)]
    ages = np.random.randint(18, 75, size=n_customers)
    # Income roughly correlated with age (peaks in middle age), plus noise
    base_income = 25000 + (np.minimum(ages, 55) - 18) * 1800
    incomes = np.clip(base_income + np.random.normal(0, 12000, n_customers), 15000, 250000)
    home_state_idx = np.random.randint(0, len(STATE_CITY_PAIRS), size=n_customers)

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_age": ages,
        "customer_income": incomes.round(2),
        "home_state_idx": home_state_idx,
    })
    return customers


def generate_merchants(n_merchants=N_MERCHANTS):
    merchant_ids = [f"MERCH{str(i).zfill(5)}" for i in range(1, n_merchants + 1)]
    categories = np.random.choice(MERCHANT_CATEGORIES, size=n_merchants)
    state_idx = np.random.randint(0, len(STATE_CITY_PAIRS), size=n_merchants)
    merchants = pd.DataFrame({
        "merchant_id": merchant_ids,
        "merchant_category": categories,
        "merchant_state_idx": state_idx,
    })
    return merchants


def generate_transactions(n_transactions=N_TRANSACTIONS):
    customers = generate_customers()
    merchants = generate_merchants()

    cust_sample = customers.sample(n=n_transactions, replace=True).reset_index(drop=True)
    merch_sample = merchants.sample(n=n_transactions, replace=True).reset_index(drop=True)

    dates = _random_dates(n_transactions)
    hours = np.random.choice(
        range(24), size=n_transactions,
        p=_hour_probabilities(),
    )

    categories = merch_sample["merchant_category"].values
    base_amounts = np.array([CATEGORY_AVG_AMOUNT[c] for c in categories])
    # Log-normal-ish spend distribution: mostly small, occasional large
    amounts = np.round(np.random.gamma(shape=2.0, scale=base_amounts / 2.0), 2)
    amounts = np.clip(amounts, 1.0, None)

    # Previous transaction amount: correlated with current spend, with noise
    previous_amounts = np.round(
        amounts * np.random.uniform(0.5, 1.5, n_transactions) + np.random.normal(0, 10, n_transactions),
        2,
    )
    previous_amounts = np.clip(previous_amounts, 1.0, None)

    is_weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates])
    is_international = np.random.choice([0, 1], size=n_transactions, p=[0.93, 0.07])

    # Distance from home: mostly small (local spend), a few far away (travel)
    distance_km = np.round(np.random.exponential(scale=15, size=n_transactions), 2)
    # International transactions realistically involve larger distances
    distance_km = np.where(
        is_international == 1,
        distance_km + np.random.uniform(500, 8000, n_transactions),
        distance_km,
    )
    distance_km = np.round(distance_km, 2)

    state_city_idx = merch_sample["merchant_state_idx"].values
    states = [STATE_CITY_PAIRS[i][0] for i in state_city_idx]
    cities = [STATE_CITY_PAIRS[i][1] for i in state_city_idx]

    payment_methods = np.random.choice(
        PAYMENT_METHODS, size=n_transactions, p=[0.35, 0.15, 0.30, 0.20]
    )
    card_types = np.random.choice(
        CARD_TYPES, size=n_transactions, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    transaction_ids = [f"TXN{str(i).zfill(7)}" for i in range(1, n_transactions + 1)]

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "transaction_date": dates,
        "customer_id": cust_sample["customer_id"].values,
        "merchant_id": merch_sample["merchant_id"].values,
        "merchant_category": categories,
        "transaction_amount": amounts,
        "city": cities,
        "state": states,
        "payment_method": payment_methods,
        "card_type": card_types,
        "customer_age": cust_sample["customer_age"].values,
        "customer_income": cust_sample["customer_income"].values,
        "transaction_hour": hours,
        "is_international": is_international,
        "is_weekend": is_weekend,
        "previous_transaction_amount": previous_amounts,
        "distance_from_home_km": distance_km,
    })

    df = _inject_unusual_transactions(df)
    df = _inject_data_quality_issues(df)

    return df


def _hour_probabilities():
    """Transactions are more likely during waking hours (8am-10pm)."""
    probs = np.ones(24)
    for h in range(24):
        if 8 <= h <= 22:
            probs[h] = 3.0
        elif 0 <= h <= 4:
            probs[h] = 0.3
    return probs / probs.sum()


def _inject_unusual_transactions(df, fraction=0.02):
    """
    Intentionally injects a small number of statistically unusual transactions
    (very large amounts, unusual hours, far from home, etc.) so the
    Isolation Forest model has real patterns to detect. These are labeled
    only as 'unusual' for our own reference (ground_truth_unusual) -- the
    model itself does NOT see this column, and the dashboard never claims
    these are confirmed fraud.
    """
    n = len(df)
    n_unusual = int(n * fraction)
    unusual_idx = np.random.choice(n, size=n_unusual, replace=False)

    df["ground_truth_unusual"] = 0
    df.loc[unusual_idx, "ground_truth_unusual"] = 1

    # Make these transactions genuinely stand out
    df.loc[unusual_idx, "transaction_amount"] = np.round(
        np.random.uniform(1500, 9000, size=n_unusual), 2
    )
    df.loc[unusual_idx, "transaction_hour"] = np.random.choice(
        [1, 2, 3, 4], size=n_unusual
    )
    df.loc[unusual_idx, "distance_from_home_km"] = np.round(
        np.random.uniform(2000, 12000, size=n_unusual), 2
    )
    df.loc[unusual_idx, "is_international"] = 1

    return df


def _inject_data_quality_issues(df, missing_fraction=0.01, dup_fraction=0.005):
    """
    Real-world transaction data is never perfectly clean. We intentionally
    inject a small number of missing values, duplicate rows, and invalid
    amounts so the cleaning/validation step in the pipeline has real work
    to do (and so the dashboard's Data Quality section has something
    meaningful to report).
    """
    n = len(df)

    # Missing values in a couple of non-critical columns
    missing_idx_city = np.random.choice(n, size=int(n * missing_fraction), replace=False)
    df.loc[missing_idx_city, "city"] = np.nan

    missing_idx_income = np.random.choice(n, size=int(n * missing_fraction), replace=False)
    df.loc[missing_idx_income, "customer_income"] = np.nan

    # A few invalid (negative or zero) transaction amounts -- data entry errors
    invalid_idx = np.random.choice(n, size=int(n * 0.003), replace=False)
    df.loc[invalid_idx, "transaction_amount"] = -abs(df.loc[invalid_idx, "transaction_amount"])

    # Duplicate a handful of rows to simulate double-recorded transactions
    n_dups = int(n * dup_fraction)
    dup_rows = df.sample(n=n_dups, random_state=RANDOM_SEED).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main():
    df = generate_transactions()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows (including injected duplicates) -> {out_path}")
    print(f"Ground-truth 'unusual' transactions injected: {df['ground_truth_unusual'].sum():,}")


if __name__ == "__main__":
    main()
