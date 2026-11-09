-- Independent checks on the cleaned data in `loans_clean`.
-- Pandas cleaned the data; these queries test the result without trusting the cleaning code.
-- Run them in pgAdmin's Query Tool, or all at once with: python db/run_checks.py

-- name: row_count
-- Must equal the number of unique loan IDs in the raw file.
SELECT COUNT(*) AS row_count FROM loans_clean;

-- name: null_counts
-- Nulls in the columns the model depends on. Every value must be 0.
SELECT
  COUNT(*) - COUNT(ltv)                 AS null_ltv,
  COUNT(*) - COUNT(bureau_score)        AS null_bureau_score,
  COUNT(*) - COUNT(disbursed_amount)    AS null_disbursed_amount,
  COUNT(*) - COUNT(age_at_disbursal)    AS null_age,
  COUNT(*) - COUNT(employment_type)     AS null_employment_type,
  COUNT(*) - COUNT(loan_default)        AS null_loan_default
FROM loans_clean;

-- name: duplicate_ids
-- Loan IDs that appear more than once. Must be 0.
SELECT COUNT(*) AS duplicate_ids
FROM (
  SELECT loan_id FROM loans_clean GROUP BY loan_id HAVING COUNT(*) > 1
) AS dupes;

-- name: value_ranges
-- Impossible values mean the cleaning missed something.
SELECT
  MIN(ltv)               AS min_ltv,
  MAX(ltv)               AS max_ltv,
  MIN(age_at_disbursal)  AS min_age,
  MAX(age_at_disbursal)  AS max_age,
  MIN(bureau_score)      AS min_bureau_score,
  MAX(bureau_score)      AS max_bureau_score,
  SUM(CASE WHEN disbursed_amount > asset_cost THEN 1 ELSE 0 END) AS loans_above_asset_cost
FROM loans_clean;

-- name: target_balance
-- Share of defaults and any target value that is not 0 or 1.
SELECT
  ROUND(CAST(AVG(loan_default) AS NUMERIC), 4)                   AS default_rate,
  SUM(CASE WHEN loan_default NOT IN (0, 1) THEN 1 ELSE 0 END)    AS invalid_target_values
FROM loans_clean;
