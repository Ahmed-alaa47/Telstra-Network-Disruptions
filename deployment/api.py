"""
FastAPI Deployment for Telstra Network Disruptions Prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional
import pickle
import os

app = FastAPI(
    title="Telstra Network Disruptions API",
    description="API for predicting network fault severity",
    version="1.0.0"
)

model = None
preprocessor = None

class PredictionRequest(BaseModel):
    location: str
    event_type: str
    log_feature: str
    resource_type: str
    severity_type: str
    volume: int
    event_count: Optional[int] = 1
    log_count: Optional[int] = 1
    resource_count: Optional[int] = 1
    severity_count: Optional[int] = 1
    
    class Config:
        schema_extra = {
            "example": {
                "location": "location_91",
                "event_type": "event_type_34",
                "log_feature": "feature_315",
                "resource_type": "resource_type_2",
                "severity_type": "severity_type_2",
                "volume": 200,
                "event_count": 3,
                "log_count": 5,
                "resource_count": 2,
                "severity_count": 4
            }
        }

class PredictionResponse(BaseModel):
    fault_severity: int
    confidence: float
    probabilities: dict
    fault_description: str

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

@app.on_event("startup")
async def load_model_and_preprocessor():
    global model, preprocessor
    
    try:
        model_path = 'outputs/models/telstra_nn_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✓ Model loaded successfully")
        else:
            print(f"✗ Model file not found at {model_path}")
            print("  Please run 'python main.py' first to train the model")
            
        preprocessor_path = 'outputs/models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("✓ Preprocessor loaded successfully")
        else:
            print(f"✗ Preprocessor file not found at {preprocessor_path}")
            print("  Please run 'python scripts/save_preprocessor.py' first")
            
    except Exception as e:
        print(f"Error loading model/preprocessor: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Telstra Network Disruptions API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None
    
    return {
        "status": "healthy" if (model_loaded and preprocessor_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Neural Network",
        "framework": "TensorFlow/Keras",
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "num_layers": len(model.layers),
        "total_params": int(model.count_params()),
        "classes": {
            0: "No Fault",
            1: "Few Faults",
            2: "Many Faults"
        }
    }

def preprocess_input(data: PredictionRequest) -> np.ndarray:
    """Preprocess input data for prediction"""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    
    # Create dataframe from input
    input_df = pd.DataFrame([{
        'location': data.location,
        'event_type_mode': data.event_type,
        'resource_type_mode': data.resource_type,
        'severity_type_mode': data.severity_type,
        'event_count': data.event_count,
        'log_count': data.log_count,
        'resource_count': data.resource_count,
        'severity_count': data.severity_count,
        'event_unique': 1,
        'log_unique': 1,
        'resource_unique': 1,
        'severity_unique': 1
    }])
    
    for col in ['location', 'event_type_mode', 'resource_type_mode', 'severity_type_mode']:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            input_df[col] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    input_df['event_log_ratio'] = input_df['event_count'] / (input_df['log_count'] + 1)
    input_df['severity_resource_ratio'] = input_df['severity_count'] / (input_df['resource_count'] + 1)
    input_df['total_features'] = input_df[['event_count', 'log_count', 'resource_count', 'severity_count']].sum(axis=1)
    input_df['total_unique'] = input_df[['event_unique', 'log_unique', 'resource_unique', 'severity_unique']].sum(axis=1)
    
    expected_features = preprocessor.get_feature_names()
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[expected_features]
    
    X = preprocessor.scaler.transform(input_df)
    
    return X

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """Make a single prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = preprocess_input(data)
        
        prediction_proba = model.predict(X, verbose=0)[0]
        prediction = int(np.argmax(prediction_proba))
        confidence = float(prediction_proba[prediction])
        
        fault_descriptions = {
            0: "No Fault - Network operating normally",
            1: "Few Faults - Minor network issues detected",
            2: "Many Faults - Severe network disruptions"
        }
        
        return PredictionResponse(
            fault_severity=prediction,
            confidence=confidence,
            probabilities={
                "no_fault": float(prediction_proba[0]),
                "few_faults": float(prediction_proba[1]),
                "many_faults": float(prediction_proba[2])
            },
            fault_description=fault_descriptions[prediction]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(data: BatchPredictionRequest):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for item in data.predictions:
            pred = await predict(item)
            results.append(pred.dict())
        
        return {
            "predictions": results,
            "count": len(results),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)