# Vehicle Loan Default Risk Predictor

A production-grade credit risk scoring and decisioning engine for vehicle loan applications (~2.33 lakh loans), built with cost-sensitive XGBoost, monotonic risk constraints, and isotonic probability calibration.

---

## Key Highlights

- **Cost-Sensitive Loss:** Penalizes defaults on larger ticket sizes (scaling weights by `disbursed_amount / mean_disbursed`).
- **Monotonic Constraints:** Enforces domain sanity into tree splits (e.g., higher LTV or more inquiries can only increase default risk, never decrease it).
- **Leak-Free Target Encoding:** Smoothed out-of-fold target encoding for high-cardinality dealership (`supplier_id`) and branch IDs.
- **Isotonic Probability Calibration:** Maps raw tree margins into true empirical default probabilities.
- **$F_2$-Optimized Decision Threshold:** Prioritizes catching costly defaults over false alarms ($F_2 = 0.595$ at optimal cutoff `0.231`).
- **Dual Serving:** FastAPI REST endpoint (`<5ms` latency) + interactive Streamlit underwriting UI.

---

## Performance Summary

| Metric | Score | Note |
|---|---|---|
| **Holdout PR-AUC** | `0.3465` | Evaluated on 21.7% imbalanced default distribution |
| **Holdout ROC-AUC** | `0.6617` | Overall discrimination ability |
| **Holdout $F_2$ Score** | `0.5951` | At optimal threshold `0.231` (weights recall $2\times$ over precision) |
| **Default Base Rate** | `21.7%` | 50,611 defaults out of 2,33,154 total applications |

---

## Project Structure

```text
├── config/
│   └── config.yaml              # Centralized data, feature, and training config
├── dataset/
│   ├── train.csv                # Raw dataset (~2.33 lakh loans)
│   └── processed/               # Normalized relational tables (loan, customer, bureau, risk)
├── src/
│   ├── config.py                # PROJECT_ROOT + config.yaml loader
│   ├── tracking.py              # Optional MLflow run logging
│   ├── data/                    # Ingestion (load.py) & shared feature building (preprocess.py)
│   ├── features/                # Smoothed TargetEncoder wrapper (encoder.py)
│   ├── training/                # OOF training (train.py), weights (weights.py), tuning (tune.py)
│   ├── evaluation/              # Metrics suite (metrics.py) & calibration plots (plots.py)
│   └── inference/               # Pipeline loader & inference engine (predict.py)
├── pipelines/
│   ├── train_pipeline.py        # End-to-end training, calibration, and export
│   └── retrain_pipeline.py      # Automated retraining check against production model
├── app/
│   ├── main.py                  # FastAPI service (POST /predict, GET /health)
│   └── streamlit_app.py         # Underwriter dashboard
├── models/
│   └── loan_default_pipeline.pkl # Exported production pipeline bundle
├── db/
│   ├── load_to_postgres.py      # Clean with Pandas, load into PostgreSQL (loans_clean)
│   ├── data_checks.sql          # SQL checks: row count, nulls, duplicates, ranges, target balance
│   └── run_checks.py            # Runs the SQL checks and fails if any result is wrong
├── notebooks/                   # EDA, data cleaning, and prototype notebooks
├── tests/                       # Unit + inference smoke tests
└── requirements.txt             # Project dependencies
```

---

## Data Quality Checks

Cleaning happens in Pandas (`src/data/clean.py`): duplicate loan IDs removed, missing employment type filled, `DD-MM-YY` birth years fixed, `"2yrs 3mon"` durations converted to months, primary + secondary bureau accounts combined, and ratio features added. The cleaned table is then loaded into PostgreSQL and tested **independently in SQL**, so a bug in the cleaning code cannot hide itself.

```powershell
$env:DB_URL = "postgresql+psycopg2://postgres:<password>@localhost:5432/loan_db"   # never commit this
python db/load_to_postgres.py      # writes table loans_clean in batches of 10,000 rows
python db/run_checks.py            # runs db/data_checks.sql, prints PASS/FAIL, exits 1 on any failure
```

| Check | Result |
|---|---|
| Rows loaded | 233,154 (matches unique `UniqueID` count in the raw file) |
| Duplicate `loan_id` | 0 (0 duplicate rows removed during cleaning) |
| Missing values filled | 7,661 `employment_type` values (3.3%), set to `Salaried` |
| Nulls in model columns | 0 |
| LTV range | 10.03 – 95.0 |
| Age range | 18 – 69 |
| Bureau score range | 0 – 890 |
| Disbursed amount above asset cost | 0 loans |
| Default rate | 21.71% |

The checks fail on bad data. Planting a negative age, an LTV of 140 and a repeated `loan_id` makes `run_checks.py` report 3 FAILs and exit with code 1.

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv loan_venv
loan_venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

### 2. Train the Pipeline
To run the full end-to-end training pipeline (preprocessing $\to$ 5-fold CV $\to$ tuning $\to$ calibration $\to$ export):
```bash
python pipelines/train_pipeline.py
```

### 3. Launch the FastAPI Service
```bash
python app/main.py
# API running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

#### Sample Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "disbursed_amount": 550000,
    "asset_cost": 720000,
    "ltv": 78.5,
    "bureau_score": 680,
    "age_at_disbursal": 32,
    "credit_history_months": 24,
    "total_account_count": 4,
    "total_active_accounts": 2,
    "total_overdue_accounts": 0,
    "total_current_balance": 180000,
    "total_sanctioned_amount": 350000,
    "total_installment_amount": 12000,
    "inquiry_count": 1,
    "overdue_ratio": 0.0,
    "credit_utilization": 0.51,
    "new_accounts_6m": 1
  }'
```

#### Sample Response
```json
{
  "probability": 0.1842,
  "decision": "APPROVE",
  "threshold": 0.2311
}
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Provides credit officers with an interactive application scoring form with 3-tier risk cards:
- **Low Risk (Approve):** Risk $< 16\%$
- **Medium Risk (Manual Review):** Risk between $16\%$ and $23.1\%$
- **High Risk (Reject):** Risk $> 23.1\%$
