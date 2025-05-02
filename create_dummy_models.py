import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

os.makedirs('models', exist_ok=True)

def create_date_model():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'year': np.random.randint(2020, 2025, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day': np.random.randint(1, 29, n_samples),
        'dayofweek': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'month_sin': np.sin(2 * np.pi * np.random.randint(1, 13, n_samples) / 12),
        'month_cos': np.cos(2 * np.pi * np.random.randint(1, 13, n_samples) / 12),
        'dayofweek_sin': np.sin(2 * np.pi * np.random.randint(0, 7, n_samples) / 7),
        'dayofweek_cos': np.cos(2 * np.pi * np.random.randint(0, 7, n_samples) / 7),
        'month_day': np.random.randint(1, 13, n_samples) * np.random.randint(1, 29, n_samples),
        'weekend_month': np.random.randint(0, 2, n_samples) * np.random.randint(1, 13, n_samples)
    })
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    y = np.random.choice([0, 1, 2, 3], n_samples)

    model.fit(X_scaled_df, y)
    
    with open('models/date_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/date_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Sucess create dummy")


def create_weather_model():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    

    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'pm10': np.random.uniform(10, 100, n_samples),
        'so2': np.random.uniform(5, 50, n_samples),
        'co': np.random.uniform(0.5, 5, n_samples),
        'o3': np.random.uniform(10, 80, n_samples),
        'no2': np.random.uniform(5, 60, n_samples),
        'max': np.random.uniform(50, 150, n_samples)
    })
    
    params = ['PM10', 'SO2', 'CO', 'O3', 'NO2']
    for i, row in X.iterrows():
        critical_param = np.random.choice(params)
        for param in params:
            X.loc[i, f'critical_{param}'] = 1 if param == critical_param else 0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    y = np.random.choice([0, 1, 2, 3], n_samples)
    
    model.fit(X_scaled_df, y)
    
    with open('models/weather_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/weather_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Succes create model 2")

if __name__ == "__main__":
    create_date_model()
    create_weather_model()
    print("Dummy models and scalers created success") 