-- Portfolio insights on `loans_clean`. Each query answers a business question and
-- links to a decision made in the ML pipeline.
-- Run one at a time in pgAdmin (select the query, press F5), or all with: python db/run_insights.py
-- default_rate_pct = defaults / loans * 100.

-- name: 01_portfolio_summary
-- How big is the book and how much money sits in defaulted loans?
SELECT
  COUNT(*)                                                           AS loans,
  SUM(loan_default)                                                  AS defaults,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct,
  SUM(disbursed_amount)                                              AS total_disbursed,
  ROUND(100.0 * SUM(CASE WHEN loan_default = 1 THEN disbursed_amount ELSE 0 END)
        / SUM(disbursed_amount), 2)                                  AS pct_of_money_in_defaults
FROM loans_clean;

-- name: 02_default_by_ltv_band
-- Does a smaller down payment (higher LTV) raise default risk?  -> monotonic constraint on ltv
SELECT
  CASE WHEN ltv < 60 THEN '1. under 60'
       WHEN ltv < 70 THEN '2. 60-70'
       WHEN ltv < 80 THEN '3. 70-80'
       WHEN ltv < 90 THEN '4. 80-90'
       ELSE               '5. 90+' END                                AS ltv_band,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1
ORDER BY 1;

-- name: 03_default_by_bureau_band
-- Is "no credit history" the same as "bad credit"?  -> treat bureau_score = 0 separately
SELECT
  CASE WHEN bureau_score = 0   THEN '1. no history (0)'
       WHEN bureau_score < 600 THEN '2. poor (1-599)'
       WHEN bureau_score < 700 THEN '3. fair (600-699)'
       WHEN bureau_score < 750 THEN '4. good (700-749)'
       ELSE                         '5. very good (750+)' END         AS bureau_band,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                 AS pct_of_portfolio,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1
ORDER BY 1;

-- name: 04_default_by_inquiries
-- Do borrowers who recently shopped for credit default more?  -> "credit hunger" features
SELECT
  CASE WHEN inquiry_count >= 3 THEN '3+' ELSE CAST(inquiry_count AS VARCHAR(2)) END AS inquiries,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1
ORDER BY 1;

-- name: 05_default_by_new_accounts_6m
-- Does opening several new accounts in 6 months signal stress?
SELECT
  CASE WHEN new_accounts_6m >= 2 THEN '2+' ELSE CAST(new_accounts_6m AS VARCHAR(2)) END AS new_accounts_6m,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1
ORDER BY 1;

-- name: 06_default_by_age_and_employment
-- Who defaults: young or old, salaried or self-employed?
SELECT
  employment_type,
  CASE WHEN age_at_disbursal < 25 THEN '1. under 25'
       WHEN age_at_disbursal < 35 THEN '2. 25-34'
       WHEN age_at_disbursal < 45 THEN '3. 35-44'
       ELSE                            '4. 45+' END                   AS age_band,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1, 2
ORDER BY 1, 2;

-- name: 07_default_by_id_documents
-- Does providing fewer ID documents mean more risk?
SELECT
  total_ids_provided                                                 AS id_documents,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1
ORDER BY 1;

-- name: 08_loan_size_vs_money_at_risk
-- Do bigger loans default as often as small ones, and how much money is in each size band?
-- -> cost-sensitive sample weights (bigger loans count more)
WITH banded AS (
  SELECT disbursed_amount, loan_default, NTILE(4) OVER (ORDER BY disbursed_amount) AS size_quartile
  FROM loans_clean
)
SELECT
  size_quartile,
  MIN(disbursed_amount)                                              AS min_amount,
  MAX(disbursed_amount)                                              AS max_amount,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct,
  ROUND(100.0 * SUM(CASE WHEN loan_default = 1 THEN disbursed_amount ELSE 0 END)
        / SUM(SUM(CASE WHEN loan_default = 1 THEN disbursed_amount ELSE 0 END)) OVER (), 1)
                                                                     AS pct_of_defaulted_money
FROM banded
GROUP BY size_quartile
ORDER BY size_quartile;

-- name: 09_default_by_month
-- Is the default rate stable over time?  -> why a date-based split is a better test
SELECT
  "Disbursal_Month"                                                  AS month,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY 1
ORDER BY 1;

-- name: 10_riskiest_dealers
-- Do some dealers send in much riskier borrowers?  -> target encoding of supplier_id
WITH portfolio AS (
  SELECT 1.0 * SUM(loan_default) / COUNT(*) AS rate FROM loans_clean
)
SELECT
  l.supplier_id,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(l.loan_default) / COUNT(*), 2)                   AS default_rate_pct,
  ROUND(CAST(1.0 * SUM(l.loan_default) / COUNT(*) / MAX(p.rate) AS NUMERIC), 2) AS times_portfolio_rate
FROM loans_clean l
CROSS JOIN portfolio p
GROUP BY l.supplier_id
HAVING COUNT(*) >= 200
ORDER BY default_rate_pct DESC
LIMIT 10;

-- name: 11_dealer_spread
-- How different are dealers from each other (only dealers with enough loans to trust)?
WITH dealer AS (
  SELECT supplier_id, COUNT(*) AS loans, 1.0 * SUM(loan_default) / COUNT(*) AS rate
  FROM loans_clean
  GROUP BY supplier_id
  HAVING COUNT(*) >= 200
)
SELECT
  COUNT(*)                                                           AS dealers_with_200_plus_loans,
  ROUND(100.0 * MIN(rate), 1)                                        AS lowest_default_rate_pct,
  ROUND(100.0 * MAX(rate), 1)                                        AS highest_default_rate_pct,
  SUM(CASE WHEN rate > 0.30 THEN 1 ELSE 0 END)                       AS dealers_above_30_pct,
  SUM(CASE WHEN rate < 0.15 THEN 1 ELSE 0 END)                       AS dealers_below_15_pct
FROM dealer;

-- name: 12_risk_stacking
-- Do risk factors add up?  Count of: LTV above 80, poor bureau score, 2+ inquiries, any overdue account.
WITH flagged AS (
  SELECT loan_default,
         (CASE WHEN ltv > 80 THEN 1 ELSE 0 END)
       + (CASE WHEN bureau_score BETWEEN 1 AND 599 THEN 1 ELSE 0 END)
       + (CASE WHEN inquiry_count >= 2 THEN 1 ELSE 0 END)
       + (CASE WHEN total_overdue_accounts >= 1 THEN 1 ELSE 0 END)   AS risk_flags
  FROM loans_clean
)
SELECT
  risk_flags,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM flagged
GROUP BY risk_flags
ORDER BY risk_flags;

-- name: 13_default_by_state
-- Are there regional differences worth investigating? (states with 5,000+ loans)
SELECT
  state_id,
  COUNT(*)                                                           AS loans,
  ROUND(100.0 * SUM(loan_default) / COUNT(*), 2)                     AS default_rate_pct
FROM loans_clean
GROUP BY state_id
HAVING COUNT(*) >= 5000
ORDER BY default_rate_pct DESC;
