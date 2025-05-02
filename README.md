# ML Model Backend API

A Flask-based backend API that serves machine learning models for:

1. Date based air quality classification
2. Weather parameter air quality classification

## Setup

### Prerequisites

- Python 3.9+
- pip
- Your .pkl model files

### Installation

1. Clone this repository
2. Place your model files in the models directory (It should be has 4 models, including a scaler model)
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the API

```
python app.py
```

The API will be available at http://localhost:5000

## API Endpoints


### Date-based Air Quality Classification

```
POST /api/predict/date
```

Request body:

```json
{
	"date": "2023-04-15"
}
```

Response:

```json
{
	"prediction": 0,
	"category": "BAIK",
	"probability": [0.85, 0.05, 0.05, 0.05]
}
```

The API automatically calculates the following features from the date:

- year, month, day
- dayofweek
- is_weekend
- month_sin, month_cos
- dayofweek_sin, dayofweek_cos
- month_day
- weekend_month

### Weather Parameter Air Quality Classification

```
POST /api/predict/weather
```

Request body:

```json
{
	"pm10": 45.2,
	"so2": 10.5,
	"co": 1.2,
	"o3": 30.1,
	"no2": 15.6
}
```

Response:

```json
{
	"prediction": 0,
	"category": "BAIK",
	"probability": [0.8, 0.1, 0.05, 0.05],
	"max_value": 45.2,
	"critical_parameters": ["PM10"]
}
```

The API automatically calculates:

- max (maximum value among all parameters)
- critical parameters (which parameter has the maximum value)

### Classification Categories

The models return one of four categories:

- 0: BAIK (Good)
- 1: TIDAK SEHAT (Unhealthy)
- 2: SEDANG (Moderate)
- 3: SANGAT TIDAK SEHAT (Very Unhealthy)