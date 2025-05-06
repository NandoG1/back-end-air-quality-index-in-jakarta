import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:5000"

# def test_health():
#     """Test the health endpoint"""
#     response = requests.get(f"{BASE_URL}/api/health")
#     print("Health Check:")
#     print(f"Status Code: {response.status_code}")
#     print(f"Response: {json.dumps(response.json(), indent=2)}")
#     print("\n" + "-"*50 + "\n")

def test_date_prediction():
    """Test the date prediction endpoint"""
  
    current_date = datetime.now().strftime("%Y-%m-%d")
    payload = {"date": current_date}
    
    response = requests.post(f"{BASE_URL}/api/predict/date", json=payload)
    print("Date Prediction:")
    print(f"Input Date: {current_date}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Category: {result['category']}")
        print(f"Probability: {result['probability']}")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "-"*50 + "\n")

def test_weather_prediction():
    """Test the weather prediction endpoint"""

    payload = {
        "pm10": 64,
        "so2": 8,
        "co": 51,
        "o3": 19,
        "no2": 15
    }
    
    response = requests.post(f"{BASE_URL}/api/predict/weather", json=payload)
    print("Weather Prediction:")
    print(f"Input Parameters: {json.dumps(payload, indent=2)}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Category: {result['category']}")
        print(f"Max Value: {result['max_value']}")
        print(f"Critical Parameters: {result['critical_parameters']}")
        print(f"Probability: {result['probability']}")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    print("Testing API endpoints...\n")
    
    try:
        # test_health()
        test_date_prediction()
        test_weather_prediction()
        
        print("All tests completed!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Flask app is running.")
    except Exception as e:
        print(f"Error during testing: {str(e)}") 