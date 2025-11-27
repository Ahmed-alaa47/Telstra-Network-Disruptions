# Telstra-Network-Disruptions
The goal of the problem is to predict Telstra network's fault severity at a time at a particular location based on the log data available. Each row in the main dataset (train.csv, test.csv) represents a location and a time point. They are identified by the "id" column, which is the key "id" used in other data files. 

Fault severity has 3 categories: 0,1,2 (0 meaning no fault, 1 meaning only a few, and 2 meaning many). 

Different types of features are extracted from log files and other sources: event_type.csv, log_feature.csv, resource_type.csv, severity_type.csv. 

Note: “severity_type” is a feature extracted from the log files (in severity_type.csv). Often this is a severity type of a warning message coming from the log. "severity_type" is categorical. It does not have an ordering. “fault_severity” is a measurement of actual reported faults from users of the network and is the target variable (in train.csv).

# How To Run
- Download Requirements file 
    pip install -r requirements.txt
    If you are using Python 3 : python3 -m pip install -r requirements.txt
- Run Fastapi server 
    uvicorn main:app --reload
# Data Samples For Testing

{
  "location": "location 91",
  "severity_type": "severity_type 2",
  "resource_type": "resource_type 2",
  "log_feature": "feature 315",
  "volume": 200,
  "event_type": "event_type 34",
  "model": "random_forest"
}

{
  "location": "location 118",
  "severity_type": "severity_type 2",
  "resource_type": "resource_type 2",
  "log_feature": "feature 312",
  "volume": 19,
  "event_type": "event_type 34",
  "model": "random_forest"
}

{
  "location": "location 1065",
  "severity_type": "severity_type 1",
  "resource_type": "resource_type 8",
  "log_feature": "feature 82",
  "volume": 11,
  "event_type": "event_type 15",
  "model": "random_forest"
}
