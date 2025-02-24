from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Load trained models
with open("obesity_model.pkl", "rb") as file:
    obesity_model = pickle.load(file)

with open("thyroid_model.pkl", "rb") as file:
    thyroid_model = pickle.load(file)

with open("diabetes_model.pkl", "rb") as file:
    diabetes_model = pickle.load(file)

# Load label encoders
with open("obesity_label_encoders.pkl", "rb") as file:
    obesity_encoders = pickle.load(file)

with open("label_encoders.pkl", "rb") as file:
    thyroid_encoders = pickle.load(file)

# Load diabetes scaler
with open("scaler.pkl", "rb") as file:
    diabetes_scaler = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define input model for Obesity Prediction
class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# Define input model for Thyroid Prediction
class ThyroidInput(BaseModel):
    age: int
    sex: str
    on_thyroxine: str
    query_on_thyroxine: str
    on_antithyroid_medication: str
    sick: str
    pregnant: str
    thyroid_surgery: str
    I131_treatment: str
    query_hypothyroid: str
    query_hyperthyroid: str
    lithium: str
    goitre: str
    tumor: str
    hypopituitary: str
    psych: str
    TSH_measured: str
    TSH: float
    T3_measured: str
    T3: float
    TT4_measured: str
    TT4: int
    T4U_measured: str
    T4U: float
    FTI_measured: str
    FTI: int
    TBG_measured: str
    TBG: str
    referral_source: str

# Define input model for Diabetes Prediction
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
async def main():
    return {"message": "Unified API for Obesity, Thyroid & Diabetes Prediction"}

@app.post("/predict/obesity/")
async def predict_obesity(data: ObesityInput):
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Apply Label Encoding
        for col, le in obesity_encoders.items():
            if col in df.columns:
                known_classes = list(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
                df[col] = le.transform(df[col])

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        features = df.to_numpy().reshape(1, -1)
        prediction = obesity_model.predict(features)[0]

        result = obesity_encoders["NObeyesdad"].inverse_transform([prediction])[0]
        return {"obesity_level": result, "error": None}
    
    except Exception as e:
        return {"obesity_level": None, "error": str(e)}

@app.post("/predict/thyroid/")
async def predict_thyroid(data: ThyroidInput):
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        for col, le in thyroid_encoders.items():
            if col in df.columns:
                known_classes = list(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
                df[col] = le.transform(df[col])

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        features = df.to_numpy().reshape(1, -1)
        prediction = thyroid_model.predict(features)[0]

        return {"thyroid_prediction": int(prediction), "error": None}
    
    except Exception as e:
        return {"thyroid_prediction": None, "error": str(e)}

@app.post("/predict/diabetes/")
async def predict_diabetes(data: DiabetesInput):
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Scale numerical features using the saved scaler
        features = diabetes_scaler.transform(df)
        
        prediction = diabetes_model.predict(features)[0]

        return {"diabetes_prediction": int(prediction), "error": None}
    
    except Exception as e:
        return {"diabetes_prediction": None, "error": str(e)}
