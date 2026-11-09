"""FastAPI service exposing /predict and /health for loan default risk scoring."""
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import PROJECT_ROOT
from src.inference.predict import load_pipeline, predict

PIPELINE_PATH = PROJECT_ROOT / "models" / "loan_default_pipeline.pkl"


@lru_cache(maxsize=1)
def get_pipeline() -> dict:
    return load_pipeline(PIPELINE_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # load once at startup instead of on the first request
    yield


app = FastAPI(title="Vehicle Loan Default Predictor", lifespan=lifespan)


class LoanApplication(BaseModel):
    branch_id: int = 0
    supplier_id: int = 0
    manufacturer_id: int = 0
    pincode_id: int = 0
    state_id: int = 0
    employee_id: int = 0
    disbursed_amount: float = 0.0
    asset_cost: float = 0.0
    ltv: float = 0.0
    bureau_score: float = 0.0
    age_at_disbursal: float = 0.0
    credit_history_months: float = 0.0
    total_account_count: float = 0.0
    total_active_accounts: float = 0.0
    total_overdue_accounts: float = 0.0
    total_current_balance: float = 0.0
    total_sanctioned_amount: float = 0.0
    total_installment_amount: float = 0.0
    inquiry_count: float = 0.0
    overdue_ratio: float = 0.0
    credit_utilization: float = 0.0
    new_accounts_6m: float = 0.0


class PredictionResponse(BaseModel):
    probability: float
    decision: str
    threshold: float


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(application: LoanApplication) -> PredictionResponse:
    result = predict(get_pipeline(), pd.DataFrame([application.model_dump()]))
    return PredictionResponse(
        probability=round(result["probability"][0], 4),
        decision=result["decision"][0],
        threshold=result["threshold"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": get_pipeline()["model_name"]}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
