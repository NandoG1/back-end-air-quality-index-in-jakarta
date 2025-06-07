from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("OPENWEATHER_API_KEY")

app = Flask(__name__)
CORS(app)  

def load_model(model_path):
    try:
        # Check if file exists first
        abs_path = os.path.abspath(model_path)
        print(f"Attempting to load model from: {model_path}")
        print(f"Absolute path: {abs_path}")
        print(f"File exists: {os.path.exists(abs_path)}")
        print(f"Current working directory: {os.getcwd()}")
        
        if not os.path.exists(abs_path):
            # Try relative to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, model_path)
            print(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                abs_path = alt_path
            else:
                print(f"Model file not found at any location: {model_path}")
                return None
        
        with open(abs_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Successfully loaded model from: {abs_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e} from {model_path}")
        return None

DATE_MODEL_PATH = "models/weather_date_model.pkl"
WEATHER_MODEL_PATH = "models/weather_parameter_model.pkl"
DATE_SCALER_PATH = "models/weather_date_scaler.pkl"
WEATHER_SCALER_PATH = "models/weather_parameter_scaler.pkl"


PREDICT_DATE_MODEL_PATH = "models/weather_date_model.pkl"
PREDICT_DATE_SCALER_PATH = "models/weather_date_scaler.pkl"

date_model = None
weather_model = None
date_scaler = None
weather_scaler = None
predict_date_model = None
predict_date_scaler = None


JAKARTA_LAT = -6.2088
JAKARTA_LON = 106.8456

def init_models():
    global date_model, weather_model, date_scaler, weather_scaler, predict_date_model, predict_date_scaler
    
    print("\nInitializing models...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # List contents of models directory
    models_dir = "models"
    if os.path.exists(models_dir):
        print(f"Contents of {models_dir} directory:")
        for file in os.listdir(models_dir):
            full_path = os.path.join(models_dir, file)
            print(f"  {file} - Size: {os.path.getsize(full_path)} bytes")
    else:
        print(f"Models directory '{models_dir}' does not exist!")
        # Try absolute path
        abs_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if os.path.exists(abs_models_dir):
            print(f"Found models directory at: {abs_models_dir}")
            print(f"Contents:")
            for file in os.listdir(abs_models_dir):
                full_path = os.path.join(abs_models_dir, file)
                print(f"  {file} - Size: {os.path.getsize(full_path)} bytes")
    
    date_model = load_model(DATE_MODEL_PATH)
    weather_model = load_model(WEATHER_MODEL_PATH)
    date_scaler = load_model(DATE_SCALER_PATH)
    weather_scaler = load_model(WEATHER_SCALER_PATH)
    
    predict_date_model = load_model(DATE_MODEL_PATH)
    predict_date_scaler = load_model(PREDICT_DATE_SCALER_PATH)
    
    if date_scaler is None:
        date_scaler = StandardScaler()
        print("Created new date_scaler")
    if weather_scaler is None:
        weather_scaler = StandardScaler()
        print("Created new weather_scaler")
    if predict_date_scaler is None:
        predict_date_scaler = StandardScaler()
        print("Created new predict_date_scaler")
        
    print("\nModel loading status:")
    print(f"date_model: {'Loaded' if date_model is not None else 'Failed'}") 
    print(f"weather_model: {'Loaded' if weather_model is not None else 'Failed'}") 
    print(f"date_scaler: {'Loaded' if date_scaler is not None else 'Failed'}") 
    print(f"weather_scaler: {'Loaded' if weather_scaler is not None else 'Failed'}") 
    print(f"predict_date_model: {'Loaded' if predict_date_model is not None else 'Failed'}") 
    print(f"predict_date_scaler: {'Loaded' if predict_date_scaler is not None else 'Failed'}")

init_models()

CATEGORY_LABELS = {
    0: "BAIK", 
    1: "SANGAT TIDAK SEHAT", 
    2: "SEDANG", 
    3: "TIDAK SEHAT"
}

def get_air_quality_trend(lat, lon, start_date, end_date, api_key):
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_timestamp,
        "end": end_timestamp,
        "appid": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            records = []
            for item in data.get('list', []):
                dt = item.get('dt', 0)
                date_obj = datetime.fromtimestamp(dt)
                components = item.get('components', {})
                
                records.append({
                    'date': date_obj.date(),
                    'datetime': date_obj,
                    'pm10': components.get('pm10', 0),
                    'o3': components.get('o3', 0),
                    'no2': components.get('no2', 0),
                    'so2': components.get('so2', 0),
                    'co': components.get('co', 0),
                })
            
            df = pd.DataFrame(records)
            
            daily_df = df.groupby('date').agg({
                'pm10': 'mean',
                'o3': 'mean',
                'no2': 'mean',
                'so2': 'mean',
                'co': 'mean',
            }).reset_index()
            
            pollutants = ['pm10', 'so2', 'co', 'o3', 'no2']
            stats = {}
            
            for pollutant in pollutants:
                series = daily_df[pollutant]
                
                slope = np.polyfit(np.arange(len(series)), series, 1)[0] if len(series) > 1 else 0
                
                stats[pollutant] = {
                    'mean': series.mean(),
                    'std': series.std() if len(series) > 1 else 0,
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'range': series.max() - series.min(),
                    'last': series.iloc[-1] if not series.empty else 0,
                    'slope': slope
                }
            
            return {
                'raw_data': data,
                'daily_data': daily_df.to_dict('records'),
                'statistics': stats
            }
        else:
            return {"error": f"API Error: {response.status_code}", "details": data}
            
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/predict/date', methods=['POST'])
def predict_from_date():
    try:
        data = request.json
        date_str = data.get('date')
        
        if not date_str:
            return jsonify({"error": "Date parameter is required"}), 400
        
        try:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, '%B %d, %Y')
                except ValueError:
                    date_obj = datetime.strptime(date_str, '%d-%m-%Y')
        except Exception as e:
            return jsonify({"error": f"Invalid date format. Please use YYYY-MM-DD, Month DD, YYYY, or DD-MM-YYYY: {str(e)}"}), 400
        
        end_date = date_obj.strftime('%Y-%m-%d')
        start_date = (date_obj - timedelta(days=7)).strftime('%Y-%m-%d')
        
        result = get_air_quality_trend(JAKARTA_LAT, JAKARTA_LON, start_date, end_date, API_KEY)
        
        if 'error' in result:
            return jsonify({"error": result['error']}), 500
        
        if 'statistics' not in result:
            return jsonify({"error": "Failed to retrieve air quality statistics"}), 500
        
        stats = result['statistics']
        
        features_dict = {}
        
        pollutant_order = ['pm10', 'so2', 'co', 'o3', 'no2']
        stat_order = ['mean', 'std', 'min', 'max', 'median', 'range', 'last', 'slope']
        
        for pollutant in pollutant_order:
            for stat in stat_order:
                feature_name = f"{stat}_{pollutant}"
                features_dict[feature_name] = stats[pollutant][stat]
        
        features_df = pd.DataFrame([features_dict])
        
        print("Feature names being used for prediction:")
        print(features_df.columns.tolist())
        
        weather_params = {
            'pm10': stats['pm10']['last'],
            'so2': stats['so2']['last'],
            'co': stats['co']['last'],
            'o3': stats['o3']['last'],
            'no2': stats['no2']['last']
        }
        
        df = pd.DataFrame([weather_params])
        df['max'] = df[['pm10', 'so2', 'co', 'o3', 'no2']].max(axis=1)
        
        for param in ['pm10', 'so2', 'co', 'o3', 'no2']:
            df[f'critical_{param.upper()}'] = (df[param] == df['max']).astype(int)
        
        critical_params = []
        for param in ['PM10', 'SO2', 'CO', 'O3', 'NO2']:
            if df[f'critical_{param}'].iloc[0] == 1:
                critical_params.append(param)
        
        if date_model is not None:
            if predict_date_scaler is not None and hasattr(predict_date_scaler, 'mean_') and predict_date_scaler.mean_ is not None:
                features_scaled = predict_date_scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
            
            prediction = date_model.predict(features_df)[0]
            prediction_proba = date_model.predict_proba(features_df)[0].tolist() if hasattr(date_model, 'predict_proba') else None
            model_used = "predict_date_model"
            print("Successfully used predict_date_model for prediction")
        elif date_model is not None:
            # Fallback to date_model if predict_date_model fails
            if date_scaler is not None and hasattr(date_scaler, 'mean_') and date_scaler.mean_ is not None:
                features_scaled = date_scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
            
            prediction = date_model.predict(features_df)[0]
            prediction_proba = date_model.predict_proba(features_df)[0].tolist() if hasattr(date_model, 'predict_proba') else None
            model_used = "date_model (fallback)"
            print("Using date_model as fallback for prediction")
        else:
            # Try to reinitialize models once more
            print("Attempting to reinitialize models...")
            init_models()
            
            if predict_date_model is not None:
                if predict_date_scaler is not None and hasattr(predict_date_scaler, 'mean_') and predict_date_scaler.mean_ is not None:
                    features_scaled = predict_date_scaler.transform(features_df)
                    features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
                
                prediction = predict_date_model.predict(features_df)[0]
                prediction_proba = predict_date_model.predict_proba(features_df)[0].tolist() if hasattr(predict_date_model, 'predict_proba') else None
                model_used = "predict_date_model (after reinit)"
                print("Successfully used predict_date_model after reinitialization")
            else:
                return jsonify({"error": "No prediction models available"}), 500
            
        pollutant_stats = {}
        for pollutant, values in stats.items():
            pollutant_stats[pollutant] = {
                'mean': float(values['mean']),
                'std': float(values['std']),
                'min': float(values['min']),
                'max': float(values['max']),
                'median': float(values['median']),
                'range': float(values['range']),
                'last': float(values['last']),
                'slope': float(values['slope'])
            }
        
        result = {
            "prediction": int(prediction),
            "category": CATEGORY_LABELS.get(int(prediction), "UNKNOWN"),
            "probability": prediction_proba,
            "max_value": float(df['max'].iloc[0]),
            "critical_parameters": critical_params,
            "model_used": model_used,
            "air_quality_statistics": pollutant_stats,
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/weather', methods=['POST'])
def predict_from_weather():
    try:
        data = request.json
        
        weather_params = {
            'pm10': data.get('pm10'),
            'so2': data.get('so2'),
            'co': data.get('co'),
            'o3': data.get('o3'),
            'no2': data.get('no2')
        }
        
        missing_params = [param for param, value in weather_params.items() if value is None]
        if missing_params:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400
        
        df = pd.DataFrame([weather_params])
        
        df['max'] = df[['pm10', 'so2', 'co', 'o3', 'no2']].max(axis=1)
        
        for param in ['pm10', 'so2', 'co', 'o3', 'no2']:
            df[f'critical_{param.upper()}'] = (df[param] == df['max']).astype(int)
        
        if weather_model is not None:
            X = df[['pm10', 'so2', 'co', 'o3', 'no2', 'max', 
                   'critical_PM10', 'critical_SO2', 'critical_CO', 'critical_O3', 'critical_NO2']]
            
            if weather_scaler is not None:
                if hasattr(weather_scaler, 'mean_') and weather_scaler.mean_ is not None:
                    X_scaled = weather_scaler.transform(X)
                else:
                    X_scaled = weather_scaler.fit_transform(X)
                
                X = pd.DataFrame(X_scaled, columns=X.columns)
            
            prediction = weather_model.predict(X)[0]
            prediction_proba = weather_model.predict_proba(X)[0].tolist() if hasattr(weather_model, 'predict_proba') else None
            
            critical_params = []
            for param in ['PM10', 'SO2', 'CO', 'O3', 'NO2']:
                if df[f'critical_{param}'].iloc[0] == 1:
                    critical_params.append(param)
            
            result = {
                "prediction": int(prediction),
                "category": CATEGORY_LABELS.get(int(prediction), "UNKNOWN"),
                "probability": prediction_proba,
                "max_value": float(df['max'].iloc[0]),
                "critical_parameters": critical_params
            }
            return jsonify(result)
        else:
            return jsonify({"error": "Model not loaded properly"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    # Check if models directory exists and list its contents
    models_dir_info = {}
    models_dir = "models"
    if os.path.exists(models_dir):
        models_dir_info["exists"] = True
        models_dir_info["contents"] = []
        for file in os.listdir(models_dir):
            full_path = os.path.join(models_dir, file)
            models_dir_info["contents"].append({
                "name": file,
                "size": os.path.getsize(full_path),
                "path": full_path
            })
    else:
        models_dir_info["exists"] = False
        models_dir_info["contents"] = []

    return jsonify({
        "status": "healthy", 
        "working_directory": os.getcwd(),
        "script_directory": os.path.dirname(os.path.abspath(__file__)),
        "models_directory": models_dir_info,
        "models_loaded": {
            "date_model": date_model is not None,
            "weather_model": weather_model is not None,
            "predict_date_model": predict_date_model is not None,
            "date_scaler": date_scaler is not None,
            "weather_scaler": weather_scaler is not None,
            "predict_date_scaler": predict_date_scaler is not None
        },
        "model_paths": {
            "DATE_MODEL_PATH": DATE_MODEL_PATH,
            "WEATHER_MODEL_PATH": WEATHER_MODEL_PATH,
            "PREDICT_DATE_MODEL_PATH": PREDICT_DATE_MODEL_PATH
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 