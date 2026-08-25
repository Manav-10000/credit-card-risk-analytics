-- ============================================================================
-- analysis.sql
-- Plain-SQL reference for the Credit Card Transaction Risk & Analytics
-- Dashboard project.
--
-- These are the exact queries used by src/analysis.py (mirrored here so you
-- can read/run them without touching any Python -- useful in interviews or
-- if you want to load transactions.db into a SQLite GUI like DB Browser).
--
-- Table: transactions
-- Loaded from the cleaned + anomaly-scored dataset. Key columns:
--   transaction_id, transaction_date, customer_id, merchant_id,
--   merchant_category, transaction_amount, city, state, payment_method,
--   card_type, customer_age, customer_income, transaction_hour,
--   is_international, is_weekend, previous_transaction_amount,
--   distance_from_home_km, anomaly_score, is_anomaly
-- ============================================================================


-- 1. Monthly transaction value & volume trend
SELECT
    strftime('%Y-%m', transaction_date) AS month,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_value,
    ROUND(AVG(transaction_amount), 2) AS avg_value
FROM transactions
GROUP BY month
ORDER BY month;


-- 2. Transaction count by merchant category
SELECT
    merchant_category,
    COUNT(*) AS transaction_count
FROM transactions
GROUP BY merchant_category
ORDER BY transaction_count DESC;


-- 3. Total & average spend by merchant category
SELECT
    merchant_category,
    ROUND(SUM(transaction_amount), 2) AS total_spend,
    ROUND(AVG(transaction_amount), 2) AS avg_spend
FROM transactions
GROUP BY merchant_category
ORDER BY total_spend DESC;


-- 4. Top 10 customers by total transaction value
SELECT
    customer_id,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_spend
FROM transactions
GROUP BY customer_id
ORDER BY total_spend DESC
LIMIT 10;


-- 5. Top 10 merchants by total transaction value
SELECT
    merchant_id,
    merchant_category,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_value
FROM transactions
GROUP BY merchant_id, merchant_category
ORDER BY total_value DESC
LIMIT 10;


-- 6. Weekend vs weekday spend comparison
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_value,
    ROUND(AVG(transaction_amount), 2) AS avg_value
FROM transactions
GROUP BY day_type;


-- 7. International vs domestic spend comparison
SELECT
    CASE WHEN is_international = 1 THEN 'International' ELSE 'Domestic' END AS txn_type,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_value,
    ROUND(AVG(transaction_amount), 2) AS avg_value
FROM transactions
GROUP BY txn_type;


-- 8. High-value transactions (>= $1,000), most recent-largest first
SELECT
    transaction_id, transaction_date, customer_id, merchant_id,
    merchant_category, transaction_amount, state, is_international
FROM transactions
WHERE transaction_amount >= 1000
ORDER BY transaction_amount DESC
LIMIT 100;


-- 9. Transaction value by state
SELECT
    state,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_value,
    ROUND(AVG(transaction_amount), 2) AS avg_value
FROM transactions
GROUP BY state
ORDER BY total_value DESC;


-- 10. (Bonus) Anomaly rate by merchant category
--     Shows which categories have a higher share of model-flagged
--     "potentially unusual" transactions -- NOT a fraud confirmation,
--     just a starting point for a risk analyst to investigate further.
SELECT
    merchant_category,
    COUNT(*) AS transaction_count,
    SUM(is_anomaly) AS flagged_count,
    ROUND(100.0 * SUM(is_anomaly) / COUNT(*), 2) AS flagged_rate_pct
FROM transactions
GROUP BY merchant_category
ORDER BY flagged_rate_pct DESC;
