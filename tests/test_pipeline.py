import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as app_main
from src.config import PROJECT_ROOT
from src.data.clean import clean_raw, duration_to_months
from src.evaluation.metrics import compute_all_metrics
from src.inference.predict import load_pipeline, predict
from src.training.weights import compute_normalized_weights

SAMPLE_APPLICATION = {
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
    "new_accounts_6m": 1,
}


class TestCleanRaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.read_csv(PROJECT_ROOT / "dataset" / "train.csv", nrows=2000)
        cls.raw = raw
        cls.clean = clean_raw(raw)

    def test_duration_to_months(self):
        self.assertEqual(duration_to_months("2yrs 3mon"), 27)
        self.assertEqual(duration_to_months("0yrs 0mon"), 0)
        self.assertEqual(duration_to_months(None), 0)

    def test_two_digit_birth_years_do_not_land_in_the_future(self):
        raw = self.raw.head(1).copy()
        raw["Date.of.Birth"] = "01-01-64"  # parses as 2064 without the century fix
        raw["DisbursalDate"] = "03-08-18"
        self.assertEqual(clean_raw(raw)["age_at_disbursal"].iloc[0], 54)

    def test_missing_employment_type_is_filled(self):
        raw = self.raw.head(3).copy()
        raw["Employment.Type"] = None
        self.assertTrue((clean_raw(raw)["employment_type"] == "Salaried").all())

    def test_duplicate_loan_ids_are_removed(self):
        doubled = pd.concat([self.raw.head(5), self.raw.head(5)])
        self.assertEqual(len(clean_raw(doubled)), 5)

    def test_no_missing_values_and_sane_ranges(self):
        self.assertEqual(self.clean.isna().sum().sum(), 0)
        self.assertTrue(self.clean["age_at_disbursal"].between(18, 75).all())
        self.assertTrue(self.clean["ltv"].between(0, 100).all())


class TestVehicleLoanPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline(PROJECT_ROOT / "models" / "loan_default_pipeline.pkl")

    def test_pipeline_loaded_properly(self):
        for key in ("model_name", "calibrated_model", "feature_columns", "optimal_f2_threshold"):
            self.assertIn(key, self.pipeline)

    def test_metrics_computation(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.25])
        metrics = compute_all_metrics(y_true, y_proba)
        for key in ("pr_auc", "roc_auc", "f2"):
            self.assertIn(key, metrics)
        self.assertGreater(metrics["roc_auc"], 0.9)

    def test_sample_weights_normalization(self):
        X = pd.DataFrame({"disbursed_amount": np.log1p([50000, 100000, 150000])})
        y = pd.Series([0, 1, 0])
        weights = compute_normalized_weights(X, y, scale_pos_weight=3.5)
        self.assertAlmostEqual(weights.mean(), 1.0, places=5)
        self.assertGreater(weights[1], weights[0])  # the default carries the extra weight

    def test_inference_prediction(self):
        sample = pd.DataFrame([{
            **SAMPLE_APPLICATION,
            "branch_id": 67, "supplier_id": 22807, "manufacturer_id": 45,
            "pincode_id": 1441, "state_id": 6, "employee_id": 100,
        }])
        res = predict(self.pipeline, sample)
        self.assertEqual(len(res["probability"]), 1)
        self.assertIn(res["decision"][0], ["APPROVE", "REJECT"])
        self.assertTrue(0.0 <= res["probability"][0] <= 1.0)
        self.assertGreater(res["threshold"], 0.0)

    def test_api_health_and_predict(self):
        self.assertEqual(app_main.health(), {"status": "ok", "model": "XGBoost"})

        resp = app_main.predict_endpoint(app_main.LoanApplication(**SAMPLE_APPLICATION))
        self.assertIn(resp.decision, ["APPROVE", "REJECT"])
        self.assertTrue(0.0 <= resp.probability <= 1.0)
        self.assertGreater(resp.threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
