from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np

# Load the trained model and feature names
try:
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    print("ERROR: Could not load model or features:", e)
    raise

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn based on numeric features.",
    version="1.0"
)

# Input schema
class ChurnRequest(BaseModel):
    age: float
    monthly_charges: float
    tenure_months: float
    num_logins_30_days: float
    num_support_tickets_90_days: float
    is_premium_plan: int
    request_id: Optional[str] = None

# Output schema
class ChurnResponse(BaseModel):
    predicted_class: int
    class_probabilities: dict
    model_version: str
    request_id: Optional[str] = None

@app.get("/")
def read_root():
    return {
        "message": "Customer Churn API is running. Use POST /predict",
        "example": {
            "age": 35,
            "monthly_charges": 90.5,
            "tenure_months": 10,
            "num_logins_30_days": 15,
            "num_support_tickets_90_days": 3,
            "is_premium_plan": 0
        }
    }

@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    # Convert the request into the correct feature vector order
    features = np.array([[
        req.age,
        req.monthly_charges,
        req.tenure_months,
        req.num_logins_30_days,
        req.num_support_tickets_90_days,
        req.is_premium_plan
    ]])

    prediction = int(model.predict(features)[0])

    # Get probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = model.classes_
        probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
    else:
        probabilities = {str(prediction): 1.0}

    return ChurnResponse(
        predicted_class=prediction,
        class_probabilities=probabilities,
        model_version="v1.0",
        request_id=req.request_id
    )
