#!/usr/bin/env python3

import requests
import json
from datetime import datetime

# Test the API endpoints
def test_health_endpoint(base_url):
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Models loaded: {data['models_loaded']}")
            print(f"Models directory exists: {data.get('models_directory', {}).get('exists', False)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_date_prediction(base_url):
    print("\nTesting date prediction endpoint...")
    try:
        test_date = "2024-01-15"
        response = requests.post(f"{base_url}/api/predict/date", 
                               json={"date": test_date},
                               headers={"Content-Type": "application/json"})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data.get('prediction')} - {data.get('category')}")
            print(f"Model used: {data.get('model_used')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing date prediction: {e}")
        return False

if __name__ == "__main__":
    # Test both local and production
    urls = [
        "http://localhost:5000",
        "https://lovely-tenderness-production.up.railway.app"
    ]
    
    for base_url in urls:
        print(f"\n{'='*50}")
        print(f"Testing: {base_url}")
        print(f"{'='*50}")
        
        health_ok = test_health_endpoint(base_url)
        if health_ok:
            test_date_prediction(base_url)
        else:
            print("Skipping prediction test due to health check failure")
