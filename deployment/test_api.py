import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

def test_root():
    """Test root endpoint"""
    print_section("Testing Root Endpoint")
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_health():
    """Test health endpoint"""
    print_section("Testing Health Endpoint")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Test model info endpoint"""
    print_section("Testing Model Info Endpoint")
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_single_prediction():
    """Test single prediction"""
    print_section("Testing Single Prediction")
    
    # Test data 1
    test_data_1 = {
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
    
    print("Test Case 1:")
    print(f"Request:\n{json.dumps(test_data_1, indent=2)}\n")
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data_1)
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    print("\n" + "-" * 80 + "\n")
    
    test_data_2 = {
        "location": "location_118",
        "event_type": "event_type_34",
        "log_feature": "feature_312",
        "resource_type": "resource_type_2",
        "severity_type": "severity_type_2",
        "volume": 19,
        "event_count": 2,
        "log_count": 3,
        "resource_count": 1,
        "severity_count": 2
    }
    
    print("Test Case 2:")
    print(f"Request:\n{json.dumps(test_data_2, indent=2)}\n")
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data_2)
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_batch_prediction():
    print_section("Testing Batch Prediction")
    
    batch_data = {
        "predictions": [
            {
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
            },
            {
                "location": "location_118",
                "event_type": "event_type_34",
                "log_feature": "feature_312",
                "resource_type": "resource_type_2",
                "severity_type": "severity_type_2",
                "volume": 19,
                "event_count": 2,
                "log_count": 3,
                "resource_count": 1,
                "severity_count": 2
            },
            {
                "location": "location_1065",
                "event_type": "event_type_15",
                "log_feature": "feature_82",
                "resource_type": "resource_type_8",
                "severity_type": "severity_type_1",
                "volume": 11,
                "event_count": 1,
                "log_count": 2,
                "resource_count": 1,
                "severity_count": 1
            }
        ]
    }
    
    print(f"Request: {len(batch_data['predictions'])} predictions\n")
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def main():
    print("\n" + "=" * 80)
    print("TELSTRA NETWORK DISRUPTIONS API - TEST SUITE")
    print("=" * 80)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print_section("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 80)
        print("ERROR: Could not connect to API server")
        print("=" * 80)
        print("\nMake sure the server is running:")
        print("  cd deployment")
        print("  uvicorn api:app --reload")
        print("\nOr run:")
        print("  python scripts/run_server.bat")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()