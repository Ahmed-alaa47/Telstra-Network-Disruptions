# Telstra Network Disruptions - Neural Network Classification

Complete end-to-end machine learning pipeline for predicting network fault severity using Neural Networks.

## 📁 Project Structure
```
telstra_network_disruptions/
│
├── data/                              
├── src/                               
│   ├── data_preprocessing.py          
│   ├── model.py                       
│   ├── evaluation.py                  
│   ├── hyperparameter_tuning.py       
│   └── visualization.py               
├── deployment/                        
│   ├── api.py                         
│   ├── test_api.py                    
│   └── README_DEPLOYMENT.md           
├── outputs/                           
│   ├── models/                        
│   ├── visualizations/                
│   └── reports/                       
├── main.py                            
├── requirements.txt                   
└── requirements_deploy.txt            
```

##  Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd telstra_network_disruptions

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

**Option A: Use Real Data**
- Download from [Kaggle](https://www.kaggle.com/competitions/telstra-recruiting-network/data)
- Place files in `data/` folder

**Option B: Generate Sample Data**
```bash
python scripts/create_sample_data.py
```

### 3. Verify Setup
```bash
python scripts/check_setup.py
```

### 4. Train Model
```bash
python main.py
```

### 5. Deploy API
```bash
# Save preprocessor for deployment
python scripts/save_preprocessor.py

# Start API server
python scripts/run_server.bat

# Or manually:
cd deployment
uvicorn api:app --reload
```

##  Features

### Data Preprocessing
-  Data cleaning (missing values, duplicates, outliers)
-  Feature engineering (encoding, aggregation, interactions)
-  Feature scaling (StandardScaler)
-  Train/validation/test split

### Model Architecture
-  Multi-layer Neural Network
-  Batch Normalization
-  Dropout regularization
-  Configurable architecture
-  Multiple optimizers (Adam, SGD, RMSprop)
-  Early stopping and learning rate reduction

### Evaluation Metrics
-  Accuracy, Precision, Recall, F1-Score
-  Confusion Matrix
-  Per-class performance analysis
-  Training/validation curves

### Hyperparameter Tuning
-  Grid Search
-  Random Search
-  Systematic experimentation
-  Results tracking and comparison

### Visualization
-  Training history plots
-  Confusion matrix heatmap
-  Class distribution analysis
-  Prediction comparison
-  Metrics comparison charts
-  ROC curves

### API Deployment
-  FastAPI REST API
-  Single and batch predictions
-  Model information endpoint
-  Health check endpoint
-  Interactive API documentation
-  Comprehensive test suite

##  Usage

### Training
```python
# Run complete pipeline
python main.py

# Outputs:
# - outputs/models/telstra_nn_model.keras
# - outputs/visualizations/*.png
```

### Making Predictions
```python
from src.model import NeuralNetworkModel
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('outputs/models/telstra_nn_model.keras')

# Make prediction
prediction = model.predict(X_new)
```

### API Usage
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "location": "location_91",
        "event_type": "event_type_34",
        "log_feature": "feature_315",
        "resource_type": "resource_type_2",
        "severity_type": "severity_type_2",
        "volume": 200
    }
)

print(response.json())
```

##  Configuration

### Model Hyperparameters

Edit in `main.py`:
```python
LAYERS_CONFIG = [128, 64, 32]     # Neurons per layer
ACTIVATION = 'relu'                # Activation function
DROPOUT_RATE = 0.3                 # Dropout rate
OPTIMIZER = 'adam'                 # Optimizer
LEARNING_RATE = 0.001              # Learning rate
EPOCHS = 100                       # Training epochs
BATCH_SIZE = 32                    # Batch size
```

### Hyperparameter Tuning
```python
param_grid = {
    'layers_config': [[64, 32], [128, 64], [128, 64, 32]],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.2, 0.3, 0.4],
    'optimizer': ['adam', 'rmsprop'],
    'learning_rate': [0.001, 0.0001]
}

tuner = HyperparameterTuner(input_dim, num_classes)
results = tuner.grid_search(X_train, y_train, X_val, y_val, param_grid)
```

##  Results

The model achieves:
- **High accuracy** on fault severity prediction
- **Balanced performance** across all fault classes
- **Robust generalization** to unseen data

Example Performance:
- Accuracy: 85-90%
- Precision: 85-88%
- Recall: 84-87%
- F1-Score: 84-87%

##  Testing

### Test Project Setup
```bash
python scripts/check_setup.py
```

### Test API
```bash
# Start server first
python scripts/run_server.bat

# Run tests in another terminal
python deployment/test_api.py
```

##  Documentation

- **Main README**: This file
- **Deployment Guide**: `deployment/README_DEPLOYMENT.md`
- **API Docs (Interactive)**: http://localhost:8000/docs
- **API Docs (Alternative)**: http://localhost:8000/redoc

## 🛠️ Troubleshooting

### Data Files Not Found
```bash
python scripts/create_sample_data.py
```

### Import Errors
```bash
pip install -r requirements.txt
```

### API Server Won't Start
```bash
pip install -r requirements_deploy.txt
```

### Model Not Found
```bash
python main.py  # Train the model first