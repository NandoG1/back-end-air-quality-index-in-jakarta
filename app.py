from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from datetime import datetime
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

DATE_MODEL_PATH = "models/date_model.pkl"
WEATHER_MODEL_PATH = "models/weather_model2.pkl"
DATE_SCALER_PATH = "models/date_scaler.pkl"
WEATHER_SCALER_PATH = "models/weather_scaler2.pkl"


date_model = None
weather_model = None
date_scaler = None
weather_scaler = None

def init_models():
    global date_model, weather_model, date_scaler, weather_scaler
    date_model = load_model(DATE_MODEL_PATH)
    weather_model = load_model(WEATHER_MODEL_PATH)
    date_scaler = load_model(DATE_SCALER_PATH)
    weather_scaler = load_model(WEATHER_SCALER_PATH)
    
    if date_scaler is None:
        date_scaler = StandardScaler()
    if weather_scaler is None:
        weather_scaler = StandardScaler()

init_models()

CATEGORY_LABELS = {
    0: "BAIK", 
    1: "SANGAT TIDAK SEHAT", 
    2: "SEDANG", 
    3: "TIDAK SEHAT"
}

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
        
        
        df = pd.DataFrame([{'tanggal': date_obj}])
        
        
        df['year'] = df['tanggal'].dt.year
        df['month'] = df['tanggal'].dt.month
        df['day'] = df['tanggal'].dt.day
        df['dayofweek'] = df['tanggal'].dt.dayofweek  
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        df['month_day'] = df['month'] * df['day']
        df['weekend_month'] = df['is_weekend'] * df['month']
        
        
        features = df.drop('tanggal', axis=1)
        
       
        if date_scaler is not None:
            
            if hasattr(date_scaler, 'mean_') and date_scaler.mean_ is not None:
                features_scaled = date_scaler.transform(features)
           
            else:
                features_scaled = date_scaler.fit_transform(features)
            
            
            features = pd.DataFrame(features_scaled, columns=features.columns)
        
        if date_model is not None:
            prediction = date_model.predict(features)[0]
            prediction_proba = date_model.predict_proba(features)[0].tolist() if hasattr(date_model, 'predict_proba') else None
            
            result = {
                "prediction": int(prediction),
                "category": CATEGORY_LABELS.get(int(prediction), "UNKNOWN"),
                "probability": prediction_proba
            }
            return jsonify(result)
        else:
            return jsonify({"error": "Model not loaded properly"}), 500
    
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

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "healthy", "models_loaded": {
#         "date_model": date_model is not None,
#         "weather_model": weather_model is not None,
#         "date_scaler": date_scaler is not None,
#         "weather_scaler": weather_scaler is not None
#     }})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 