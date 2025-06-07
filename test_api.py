#!/usr/bin/env python3
"""
Health Check Script for Flask API
Run this script to check if your Flask API and models are working properly.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5000"  # Change this if your API runs on different host/port
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"
PREDICT_DATE_ENDPOINT = f"{API_BASE_URL}/api/predict/date"
PREDICT_WEATHER_ENDPOINT = f"{API_BASE_URL}/api/predict/weather"

def print_separator(title=""):
    print("=" * 60)
    if title:
        print(f" {title} ")
        print("=" * 60)

def check_health():
    """Check the health status of the API"""
    print_separator("HEALTH CHECK")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API is running!")
            print(f"Status: {data.get('status', 'unknown')}")
            
            print("\n📊 Model Loading Status:")
            models = data.get('models_loaded', {})
            for model_name, is_loaded in models.items():
                status = "✅ Loaded" if is_loaded else "❌ Failed"
                print(f"  {model_name}: {status}")
            
            print("\n📁 Model Paths:")
            paths = data.get('model_paths', {})
            for path_name, path_value in paths.items():
                print(f"  {path_name}: {path_value}")
            
            return True, data
            
        else:
            print(f"❌ API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure your Flask app is running!")
        print(f"Trying to connect to: {HEALTH_ENDPOINT}")
        return False, None
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False, None
    except Exception as e:
        print(f"❌ Error checking health: {str(e)}")
        return False, None

def test_date_prediction():
    """Test the date prediction endpoint"""
    print_separator("DATE PREDICTION TEST")
    
    test_date = "2024-01-15"  # Use a test date
    payload = {"date": test_date}
    
    try:
        print(f"Testing date prediction with date: {test_date}")
        response = requests.post(PREDICT_DATE_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Date prediction successful!")
            print(f"Prediction: {data.get('prediction')}")
            print(f"Category: {data.get('category')}")
            print(f"Model used: {data.get('model_used')}")
            
            if 'probability' in data and data['probability']:
                print("Probability distribution:")
                for i, prob in enumerate(data['probability']):
                    print(f"  Class {i}: {prob:.4f}")
            
            return True
        else:
            print(f"❌ Date prediction failed with status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing date prediction: {str(e)}")
        return False

def test_weather_prediction():
    """Test the weather prediction endpoint"""
    print_separator("WEATHER PREDICTION TEST")
    
    # Sample weather data
    test_weather = {
        "pm10": 50.5,
        "so2": 10.2,
        "co": 1200.0,
        "o3": 80.3,
        "no2": 25.7
    }
    
    try:
        print("Testing weather prediction with sample data:")
        for param, value in test_weather.items():
            print(f"  {param}: {value}")
            
        response = requests.post(PREDICT_WEATHER_ENDPOINT, json=test_weather, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Weather prediction successful!")
            print(f"Prediction: {data.get('prediction')}")
            print(f"Category: {data.get('category')}")
            print(f"Max value: {data.get('max_value')}")
            print(f"Critical parameters: {data.get('critical_parameters')}")
            
            if 'probability' in data and data['probability']:
                print("Probability distribution:")
                for i, prob in enumerate(data['probability']):
                    print(f"  Class {i}: {prob:.4f}")
            
            return True
        else:
            print(f"❌ Weather prediction failed with status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing weather prediction: {str(e)}")
        return False

def main():
    """Main function to run all tests"""
    print(f"🔍 API Health Check Script")
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check health first
    health_ok, health_data = check_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Please make sure your Flask app is running.")
        print("Start your Flask app with: python your_flask_app.py")
        sys.exit(1)
    
    print()
    
    # Check if any models are loaded
    models_loaded = health_data.get('models_loaded', {})
    any_model_loaded = any(models_loaded.values())
    
    if not any_model_loaded:
        print("⚠️  WARNING: No models are loaded! Check your model files.")
    
    # Test predictions
    print()
    date_test_ok = test_date_prediction()
    
    print()
    weather_test_ok = test_weather_prediction()
    
    # Summary
    print_separator("SUMMARY")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Date Prediction: {'✅ PASS' if date_test_ok else '❌ FAIL'}")
    print(f"Weather Prediction: {'✅ PASS' if weather_test_ok else '❌ FAIL'}")
    
    if health_ok and date_test_ok and weather_test_ok:
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        
        if not any_model_loaded:
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("1. Make sure your model files exist in the 'models/' directory")
            print("2. Check file paths in your Flask app")
            print("3. Verify model files are not corrupted")
            print("4. Check file permissions")

if __name__ == "__main__":
    main()