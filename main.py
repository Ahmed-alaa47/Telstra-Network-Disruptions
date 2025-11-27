# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Literal

# ------------------- Load Models & Encoders -------------------
MODEL_DIR = "models"

rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
knn_model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

# Mapping for readable model names
MODELS = {
    "random_forest": rf_model,
    "svm": svm_model,
    "knn": knn_model
}

# ------------------- Pydantic Request Model -------------------
class PredictionRequest(BaseModel):
    location: str          # e.g., "location 118"
    severity_type: str     # e.g., "severity_type 2"
    resource_type: str     # e.g., "resource_type 2"
    log_feature: str       # e.g., "feature 312"
    volume: int            # positive integer
    event_type: str        # e.g., "event_type 34"
    model: Literal["random_forest", "svm", "knn"] = "random_forest"

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="Telstra Network Fault Severity Prediction API",
    description="Predict fault severity (0 = No fault, 1 = Minor, 2 = Major) using trained ML models.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Telstra Fault Severity Prediction API is running!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Create input DataFrame
        input_data = {
            "location": [request.location],
            "severity_type": [request.severity_type],
            "resource_type": [request.resource_type],
            "log_feature": [request.log_feature],
            "volume": [request.volume],
            "event_type": [request.event_type]
        }
        df = pd.DataFrame(input_data)

        # Apply the same LabelEncoder transformations as in training
        cat_columns = ["location", "severity_type", "resource_type", "log_feature", "event_type"]
        for col in cat_columns:
            le = label_encoders[col]
            # Handle unseen categories gracefully
            df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])  # fallback to first class
            df[col] = le.transform(df[col])

        # Select model
        model = MODELS[request.model]

        # Predict
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        prob_dict = {
            "predict_0_no_fault": float(probabilities[0]),
            "predict_1_minor_fault": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            "predict_2_major_fault": float(probabilities[2]) if len(probabilities) > 2 else 0.0
        }

        return {
            "fault_severity": prediction,
            "confidence_probabilities": prob_dict,
            "model_used": request.model
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}