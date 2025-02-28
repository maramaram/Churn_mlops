from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Initialize FastAPI app
app = FastAPI()

# ✅ Added a root endpoint to fix the "Not Found" error
@app.get("/")
def home():
    return {"message": "FastAPI is running! Use /predict to make predictions."}

# Request model for predictions
class PredictionInput(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

# Prediction route
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert input data to a dictionary
        input_data = data.dict()
        
        # Extract State (keeping it without encoding)
        state = input_data.pop("State")
        
        # Convert categorical variables to numerical
        input_data['International_plan'] = 1 if input_data['International_plan'] == 'Yes' else 0
        input_data['Voice_mail_plan'] = 1 if input_data['Voice_mail_plan'] == 'Yes' else 0

        # Convert to a numpy array and reshape for scaling
        feature_values = list(input_data.values())
        features_array = np.array(feature_values).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)
        return {"state": state, "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

