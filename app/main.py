"""FastAPI app exposing a /predict endpoint for loan default risk scoring."""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import load_pipeline, predict

app = FastAPI(title="Vehicle Loan Default Predictor")

# Load pipeline once at startup
PIPELINE_PATH = PROJECT_ROOT / "models" / "loan_default_pipeline.pkl"
pipeline = None


@app.on_event("startup")
def startup():
    global pipeline
    pipeline = load_pipeline(PIPELINE_PATH)
    print(f"Loaded {pipeline['model_name']} pipeline ({len(pipeline['feature_columns'])} features)")


class LoanApplication(BaseModel):
    """Fields expected from a single loan application."""
    branch_id: int = 0
    supplier_id: int = 0
    manufacturer_id: int = 0
    pincode_id: int = 0
    state_id: int = 0
    employee_id: int = 0
    disbursed_amount: float = 0
    asset_cost: float = 0
    ltv: float = 0
    bureau_score: float = 0
    age_at_disbursal: float = 0
    credit_history_months: float = 0
    total_account_count: float = 0
    total_active_accounts: float = 0
    total_overdue_accounts: float = 0
    total_current_balance: float = 0
    total_sanctioned_amount: float = 0
    total_installment_amount: float = 0
    inquiry_count: float = 0
    overdue_ratio: float = 0
    credit_utilization: float = 0
    new_accounts_6m: float = 0


class PredictionResponse(BaseModel):
    probability: float
    decision: str
    threshold: float


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(application: LoanApplication):
    row = pd.DataFrame([application.model_dump()])
    result = predict(pipeline, row)
    return PredictionResponse(
        probability=round(result["probability"][0], 4),
        decision=result["decision"][0],
        threshold=result["threshold"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": pipeline["model_name"] if pipeline else "not loaded"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
