# Vehicle Loan Default Risk Engine — Interview Revision Guide

## 1. Why I Built This
In vehicle financing across Indian NBFCs and banks, default rates hover around 20–22%. When an applicant defaults on a vehicle loan, repossessing the vehicle, auctioning it at a depreciated value, and recovering legal costs takes months and bleeds capital.

Most tutorials and Kaggle baselines treat this as a toy binary classification problem: they run a train-test split, slap standard class weights or SMOTE onto XGBoost, and report ~80% accuracy. That is meaningless in real underwriting:
1. **Accuracy is a deceptive metric**: In a dataset with ~21.7% defaults, a naive model predicting "approved" for every single borrower gets ~78.3% accuracy while driving the lending institution into insolvency.
2. **Missing large loans costs exponentially more**: Missing a default on an ₹8 lakh commercial vehicle loan is far more damaging than a default on a ₹40,000 two-wheeler loan. Standard loss functions treat every row equally.
3. **First-time borrowers have no credit history**: A massive chunk of Indian vehicle borrowers are "New to Credit" (no bureau history, bureau score = 0). Models must distinguish between a genuine clean slate and high-risk distress.
4. **Uncalibrated probabilities ruin loan pricing**: Raw tree probabilities do not reflect true statistical risk. If a credit underwriter wants to price a loan or set reserve capital, a model predicting 0.35 must actually mean that 35 out of 100 similar borrowers default.

I built this project to implement a production-grade, cost-sensitive, and calibrated risk scoring engine with strict business monotonic constraints and dual-serving interfaces (FastAPI + Streamlit).

---

## 2. What Is the Project About?
An end-to-end risk decisioning pipeline for vehicle loan origination. It processes raw applicant records (~2.33 lakh loans), normalizes them across relational entities, trains gradient-boosted trees under monotonic risk guardrails, calibrates output probabilities with isotonic regression, and determines optimal loan decisions (Approve, Manual Review, Reject) using an $F_2$-optimized threshold that penalizes missed defaults.

---

## 3. What the Dataset Tells Us & How It Shaped the ML Design

### Part A: What the Data Actually Reveals (The Portfolio Reality)
The raw dataset contains **233,154 loan records** (~2.33 lakh rows) across 41 initial fields. Analyzing it through exploratory data analysis revealed specific financial dynamics that completely dictated how the ML pipeline had to be built:

1. **Class Distribution & Portfolio Default Rate (~21.7%)**:
   - Out of ~2.33 lakh borrowers, **50,611 defaulted** while 182,543 paid on time.
   - A naive classifier that blindly approves everyone achieves ~78.3% accuracy but ruins the lending book.
2. **Loan-to-Value (LTV) Risk Inflection (>80%)**:
   - The bulk of loan applications cluster between 75% and 85% LTV.
   - Borrowers with LTV $> 80\%$ exhibit sharp default increases. When a borrower puts down less than 15–20% upfront equity (down payment), their financial stake in keeping the vehicle is low, creating high incentive to abandon the asset during distress.
3. **The "New to Credit" (NTC) Dichotomy**:
   - Nearly **48% of applicants have zero credit bureau history** (`bureau_score = 0` / "No Bureau History available").
   - Crucial finding: NTC borrowers have a baseline default rate (~20%), which is actually *lower* than borrowers with poor bureau scores (<600, where defaults exceed 30%). Bureau score is highly discriminative when credit history exists, but treating score $= 0$ as "worst score" is mathematically and commercially wrong.
4. **Credit Hunger & Velocity Spikes (Leading Stress Indicators)**:
   - Applicants with multiple bureau inquiries in the prior 6 months (`inquiry_count`) or sudden bursts of newly opened accounts (`new_accounts_6m`) showed severe default spikes.
   - In retail credit, a borrower rapidly applying for loans across multiple institutions indicates an active liquidity crunch ("credit hunger").
5. **Loan Exposure Disparity (₹13,000 to ₹9.9 Lakh)**:
   - Disbursed amounts range from micro two-wheeler loans (~₹13,000) to heavy commercial vehicles (~₹9.9 lakh), with a median around ₹53,000–₹60,000.
   - The financial loss from a bad ₹8 lakh loan cannot be treated the same as a bad ₹40,000 loan.
6. **Dealer & Branch Sourcing Hazard**:
   - Aggregating defaults by `supplier_id` (auto dealerships) and `branch_id` revealed high-risk pockets where default rates exceeded 35%–40%, compared to a healthy 12%–15% at well-managed dealerships.
   - This points to dealer-level moral hazard: certain dealers push subprime borrowers through lenient verification to meet sales targets.

---

### Part B: How That Data Story Dictated My ML Design

| Data Observation | Why It Mattered | ML Engineering Decision |
|---|---|---|
| **LTV, Bureau Score, & Inquiries have clear economic directions** | Trees can overfit in sparse leaves, creating counter-intuitive splits (e.g. approving 92% LTV over 80% LTV). | **Enforced Monotonic Constraints** into XGBoost: LTV ($+1$), Bureau Score ($-1$), Overdue Ratio ($+1$), Inquiries ($+1$). |
| **Loan tickets range from ₹13k to ₹9.9 lakh** | Unweighted loss treats all rows equally. Defaulting on an ₹8 lakh loan destroys margins far more than a ₹40,000 loan. | **Cost-Sensitive Sample Weights**: Scaled row weights by loan size relative to mean disbursement ($\text{clip}[0.5, 3.0]$) $\times$ class imbalance, normalized to $\mu = 1.0$. |
| **Separate Primary & Secondary accounts (`PRI.*`, `SEC.*`) caused collinearity** | Fragmented account counts dilute signal and increase feature dimensionality unnecessarily. | **Consolidated Total Exposure & Engineered Stress Ratios**: Created unified totals and dimensionless ratios (`installment_burden`, `overdue_per_account`, `balance_to_sanctioned`, `inquiry_per_month`). |
| **Dealers (`supplier_id`) carry major risk signal but have hundreds of categories** | One-Hot Encoding explodes memory and overfits small volume dealers; Label Encoding imposes fake ordinal ordering. | **Smoothed Out-of-Fold Target Encoding**: Used m-estimate smoothing ($m=10$, min leaf $=20$) fit strictly inside each CV fold to capture dealership quality without label leakage. |
| **Class imbalance (21.7%) and high cost of False Negatives** | The standard 0.5 classification threshold minimizes balanced error, not credit write-offs. | **$F_2$ Metric Sweeping & Isotonic Calibration**: Calibrated probabilities to match empirical default frequencies, then selected the threshold maximizing recall over precision ($F_2$ peak at **0.231**). |

---

## 4. What It Does (Core Features)
- **Multi-Source Data Ingestion**: Merges loan parameters, applicant demographics, credit bureau histories, and financial risk aggregates into a single analytical record.
- **Cost-Sensitive Sample Weighting**: Dynamically scales training weights based on loan ticket size ($\text{disbursed\_amount}$) combined with class imbalance, so the loss function heavily penalizes high-exposure defaults.
- **Monotonic Tree Constraints**: Enforces domain sanity into tree splits. For example, higher Loan-to-Value (LTV) or higher overdue accounts can *only* increase or maintain risk, never reduce it.
- **Leakage-Free High-Cardinality Encoding**: Uses smoothed out-of-fold target encoding for branch IDs, dealership/supplier IDs, manufacturer IDs, and pin codes.
- **Isotonic Probability Calibration**: Post-processes tree margins into reliable, true empirical probabilities.
- **Utility-Driven Decisioning ($F_2$-Score Thresholding)**: Sweeps precision-recall curves to find the cut-off that prioritizes recall over precision, reflecting the financial reality that catching bad loans matters twice as much as approving marginal ones.
- **Dual Serving**:
  - **FastAPI Service**: Low-latency `/predict` endpoint validated with Pydantic schemas for automated Loan Origination Systems (LOS).
  - **Streamlit Underwriter Dashboard**: Interactive UI for credit officers displaying 3-tier risk categories (Low Risk / Medium Risk Review / High Risk Reject) and financial stress ratios.
- **Automated Retraining Pipeline**: Checks new incoming batches, evaluates against the active production model, and promotes only if holdout PR-AUC improves.

---

## 5. How It Works End-to-End

```
Raw Data (train.csv: ~2.33 lakh rows)
  │
  ├─► Data Cleaning & Partitioning (notebooks/data_analysis.ipynb)
  │     • Fixed 100-year DOB parsing bug (proper age 18–69)
  │     • Parsed duration strings ("2yrs 3mon" -> 27 months)
  │     • Normalized into 4 relational CSVs (loan_details, customer_profile, bureau_history, engineered_risk)
  │
  ├─► Pipeline Ingestion (src/data/load.py & src/data/preprocess.py)
  │     • Merged on customer_id + extracted temporal flags (quarter, month, month-end)
  │     • Engineered financial ratios: installment burden, overdue per account, balance-to-sanctioned
  │     • log1p transformed skewed currency balances
  │
  ├─► 5-Fold Stratified Cross-Validation (src/training/train.py)
  │     • Nested 80/20 train-calibration split inside each fold (zero leakage)
  │     • Target-encoded high-cardinality IDs with m-estimate smoothing
  │     • Computed exposure-scaled, normalized sample weights (src/training/weights.py)
  │     • Trained base models (XGBoost, LightGBM, CatBoost, RF, ExtraTrees, GBDT)
  │     • Calibrated fold models via Isotonic Regression
  │
  ├─► Hyperparameter Tuning (src/training/tune.py)
  │     • 30 iterations of RandomizedSearchCV optimizing PR-AUC
  │     • Monotonic constraints enforced during tree building
  │
  ├─► Final Model Export (pipelines/train_pipeline.py)
  │     • Final model: Tuned XGBoost (Holdout PR-AUC ~0.346, ROC-AUC ~0.662, F2 ~0.595)
  │     • Calibrated using held-out calibration split
  │     • Optimal decision threshold locked at ~0.231
  │     • Exported to models/loan_default_pipeline.pkl + logged to MLflow
  │
  └─► Serving Layer
        • app/main.py: FastAPI REST endpoint
        • app/streamlit_app.py: Credit underwriter risk assessment UI
```

---

## 6. Tech Stack

| Component | Technology | Why Used |
|---|---|---|
| **Data Processing** | Python, Pandas, NumPy | Vectorized operations for ratio feature engineering and data cleaning across ~2.33 lakh rows. |
| **Categorical Encoding** | `category_encoders` (TargetEncoder) | Handles high-cardinality geographic/dealer IDs (`branch_id`, `supplier_id`, `pincode_id`) with m-estimate smoothing without memory bloat of one-hot encoding. |
| **Core Models** | XGBoost, LightGBM, CatBoost | Gradient boosted trees natively support missing values, non-linear feature interactions, and tree monotonic constraints. |
| **Optimization** | SciPy, Scikit-learn (`precision_recall_curve`, `StratifiedKFold`) | Threshold sweeping over $F_2$ utility curve and rigorous leak-free nested cross-validation. |
| **Probability Calibration** | Scikit-learn (`CalibratedClassifierCV`, `IsotonicRegression`) | Transforms uncalibrated boosted margins into reliable true risk probabilities. |
| **Experiment Tracking** | MLflow (`sqlite:///mlflow.db`) | Tracks hyperparameter search runs, metrics, reliability diagrams, and model artifacts reproducibly. |
| **Backend API** | FastAPI, Uvicorn, Pydantic | High-performance async REST framework with strict schema validation at the ingestion boundary. |
| **Dashboard UI** | Streamlit | Rapid interactive interface enabling credit underwriters to simulate risk scores in real-time. |

---

## 7. Smart Engineering Decisions Under the Hood

### 1. Monotonic Constraints
In credit underwriting, tree algorithms can occasionally find spurious splits where a borrower with 90% LTV appears safer than one with 75% LTV simply because of sample noise in a leaf. In production, this can lead to audit failures.
I enforced strict monotonic constraints into XGBoost:
- $\text{LTV} \to +1$ (risk must not decrease as loan-to-value rises)
- $\text{Bureau Score} \to -1$ (risk must not increase as bureau score rises)
- $\text{Overdue Ratio} \to +1$, $\text{Credit Utilization} \to +1$, $\text{Inquiry Count} \to +1$

### 2. Cost-Sensitive Loss via Financial Exposure Weights
Standard class weights (`scale_pos_weight = 4.0`) treat all defaults identically. In vehicle loans, defaulting on ₹8 lakh ruins portfolio profitability, whereas defaulting on ₹40,000 is manageable.
I designed a custom sample weighting function:
$$\text{Weight}_i = \left(\text{base\_class\_weight}_i\right) \times \text{clip}\left(\frac{\text{Disbursed Amount}_i}{\text{Mean Disbursed}}, 0.5, 3.0\right)$$
The resulting vector is normalized back to $\mu = 1.0$ to ensure gradients do not explode or drift during boosting.

### 3. Leak-Free Nested Calibration
Fitting probability calibration directly on cross-validation predictions or on the training set causes massive optimism bias.
Inside each fold of the 5-fold CV, I implemented a nested 80/20 train/calibration split:
1. TargetEncoder and XGBoost are fit solely on the 80% training split.
2. Isotonic regression is fit solely on the 20% calibration split.
3. Out-of-fold validation data is evaluated on the calibrated pipeline.

### 4. $F_2$ Metric & Optimal Thresholding
In lending, **False Negatives** (approving a borrower who defaults) lose substantial principal balance. **False Positives** (rejecting a creditworthy borrower) merely lose interest margin. A False Negative is roughly $3\times$ to $5\times$ more costly than a False Positive.
Instead of relying on the default $0.5$ cutoff, I sweep the precision-recall curve to maximize the $F_2$ score:
$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
The optimal threshold landed at **0.231**. Borrowers with calibrated risk above 23.1% are rejected or sent to manual underwriting.

---

## 8. Real Challenges Faced & How I Solved Them

1. **The 100-Year Date Parsing Bug (`Date.of.Birth`)**:
   - `pd.to_datetime(format='%d-%m-%y')` parsed two-digit years like `"68"` as `2068`.
   - *Fix*: Created a vectorized correction subtracting 100 years from future dates, establishing valid borrower ages (18 to 69 years).
2. **Messy Duration Strings in Bureau History**:
   - Features like `AVERAGE.ACCT.AGE` and `CREDIT.HISTORY.LENGTH` were stored as string formats like `"1yrs 11mon"` and `"0yrs 0mon"`.
   - *Fix*: Wrote regex/string parsing functions to convert all durations into total integer months for continuous numeric consumption.
3. **High-Cardinality Categorical Overfitting**:
   - `supplier_id` (dealerships) and `employee_id` contain hundreds of unique IDs. One-hot encoding creates extreme sparsity, while label encoding introduces arbitrary ordinality.
   - *Fix*: Implemented smoothed Target Encoding with minimum leaf sample requirements ($m=10$, leaf threshold $=20$).
4. **Scikit-Learn Version Compatibility with Calibration**:
   - Newer Scikit-Learn versions deprecated `cv='prefit'` in `CalibratedClassifierCV` in favor of `FrozenEstimator`.
   - *Fix*: Added a defensive fallback check (`try ... import FrozenEstimator`) so the code runs seamlessly across both older and modern environments.

---

## 9. How I'd Scale It in Enterprise Production

1. **Feature Store Integration (Feast / Hopsworks)**:
   - Split features into slow-moving customer profiles (updated monthly) and real-time loan application parameters (passed via API payload) to ensure offline-online feature parity.
2. **Asynchronous Batch & Stream Processing (Kafka + Celery / FastAPI)**:
   - For dealership networks submitting thousands of applications during festival sales, route requests through a Kafka queue with worker pools to prevent API timeouts.
3. **Data & Concept Drift Monitoring (Evidently / Prometheus)**:
   - Monitor Population Stability Index (PSI) and Wasserstein Distance on core risk drivers (LTV, bureau scores, inquiry spikes).
   - Track 30-day delinquency vintage curves to detect performance drift early.
4. **Containerization & Deployment**:
   - Package FastAPI into lightweight Docker containers orchestrated via Kubernetes (EKS/GKE) with horizontal pod autoscaling based on CPU and request latency.

---

## 10. Key Learnings
- **Credit ML is an economics problem, not an accuracy race**: Optimization metrics must mirror the financial profit-and-loss sheet (recall-heavy $F_2$, cost-sensitive sample weighting).
- **Domain constraints make models reliable**: Monotonic constraints are essential for preventing non-intuitive edge cases from slipping past risk committees.
- **Probabilities must mean what they say**: Boosting models give ranked scores, not true probabilities. Probability calibration is essential if model outputs feed into financial provisioning.

---

## 11. Why Not Just Use an LLM (ChatGPT or Claude)?
- **Tabular Superiority & Efficiency**: LLMs are autoregressive text models; they are fundamentally unsuited for tabular numeric calculations and non-linear feature interactions across 40+ numerical ratios.
- **Latency & Cost**: XGBoost scores an application in under 5 milliseconds on a lightweight CPU. An LLM call takes 1–3 seconds, costs money per API invocation, and cannot meet real-time loan origination SLAs.
- **Auditability & Explainability**: Under RBI and financial fair-lending regulations, lending models must be auditable, deterministic, and explainable (via SHAP and feature importance). LLMs hallucinate, lack mathematical monotonicity guarantees, and function as black boxes.

---

## 12. Moving from Basic Classification to an Advanced Banking Architecture
If I were taking this project to a Tier-1 fintech or banking environment, I would expand beyond binary classification into an **Expected Loss (EL)** and **Risk-Based Pricing** framework:

1. **Full Basel / IFRS-9 Expected Loss Framework**:
   $$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$
   - **PD (Probability of Default)**: The current calibrated model.
   - **LGD (Loss Given Default)**: A secondary regression model predicting how much money is actually recovered after vehicle repossession and auction, factoring in vehicle depreciation, vehicle model, and resale market liquidity.
   - **EAD (Exposure at Default)**: Modeling the remaining principal balance at the time default occurs.
2. **Risk-Based Dynamic Pricing Engine**:
   - Instead of a blunt binary Approve/Reject decision, calculate an annualized risk premium:
     $$\text{Interest Rate} = \text{Cost of Funds} + \text{Operating Cost} + \text{Expected Loss Margin} + \text{Profit}$$
   - Offer borderline applicants higher interest rates or require a higher down payment (lower LTV) to bring their expected risk within acceptable thresholds.
3. **Survival Analysis / Time-to-Default (Cox Proportional Hazards)**:
   - Predict *when* default is likely to happen across the loan tenure. Defaults occurring in months 1–6 indicate dealership origination fraud; defaults occurring in months 24–36 indicate macroeconomic stress.
4. **Prescriptive Uplift Modeling**:
   - Model which delinquent borrowers will self-cure versus which borrowers require early intervention (automated SMS reminders, restructured EMI plans).

