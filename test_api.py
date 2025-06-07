import requests
import json
from datetime import datetime, timedelta

# Configuration
API_URL = "http://localhost:5000/api/predict/date"  # Change if your API is hosted elsewhere
TEST_DATES = [
    datetime.now().strftime("%Y-%m-%d"),  # Today
    (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),  # A week ago
]

def test_date_prediction(date_str):
    """Test the date prediction endpoint with a specific date"""
    
    print(f"\n=== Testing with date: {date_str} ===")
    
    # Prepare the request payload
    payload = {"date": date_str}
    
    try:
        # Make the API request
        response = requests.post(API_URL, json=payload)
        
        # Print the response status
        print(f"Status Code: {response.status_code}")
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Pretty print the response
            print("Response:")
            print(json.dumps(result, indent=2))
            
            # Extract and display key information
            print("\nKey Information:")
            print(f"Prediction Category: {result.get('category')}")
            print(f"Model Used: {result.get('model_used')}")
            
            # Critical parameters
            critical_params = result.get('critical_parameters', [])
            print(f"Critical Parameters: {', '.join(critical_params) if critical_params else 'None'}")
            
            # Analysis period
            analysis_period = result.get('analysis_period', {})
            if analysis_period:
                print(f"Analysis Period: {analysis_period.get('start_date')} to {analysis_period.get('end_date')}")
                
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

def run_all_tests():
    """Run tests with all predefined dates"""
    print("Starting API tests...")
    
    for date_str in TEST_DATES:
        test_date_prediction(date_str)
        
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Option 1: Run all predefined tests
    run_all_tests()
    
    # Option 2: Test with a specific date input from user
    # custom_date = input("\nEnter a date to test (YYYY-MM-DD, Month DD, YYYY, or DD-MM-YYYY): ")
    # test_date_prediction(custom_date)