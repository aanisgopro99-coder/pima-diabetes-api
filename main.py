from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field,ConfigDict
import pandas as pd 
import numpy as np
import joblib 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "model.joblib")
threshold = joblib.load(BASE_DIR / "models" / "threshold.joblib")

app=FastAPI()

class PatientData(BaseModel):
    Pregnancies:int
    Glucose:float 
    BloodPressure:float = Field(gt=0)
    SkinThickness:float
    Insulin:float 
    Age:int=Field(gt=0,lt=120) 
    Waight:float=Field(gt=30,lt=250)
    Hight:float=Field(gt=60,lt=250)
    DiabetesPedigreeFunction:float
    model_config=ConfigDict(extra='forbid')



@app.get("/health")
def check_health():
    return{"status":"API is running"}

@app.post("/predict")
def predict(data: PatientData):
        BMI=calculate_BMI(data)
        complete_dict=integrate_BMI(data,BMI)
        df=create_df(complete_dict)
        try:
            probability = model.predict_proba(df)[0][1]
            prediction = int(probability >= threshold)

        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Please try again."
            )
        return{
            "data_message":"data received sucessfully!",
            "BMI_message":"BMI calculated sucessfully!",
            "data_ready":complete_dict,
            "prediction_status:":"ready for prediction",
            "features_status:":"features ready!",
            "features":df.to_dict(orient="records"),
            "prediction_status":"ready",
            "prediction":prediction

        }   
    



def calculate_BMI(data):
    BMI=data.Waight/(data.Hight/100)**2
    return BMI

def integrate_BMI(data,BMI):
    return{
        "Pregnancies":data.Pregnancies,
        "Glucose":data.Glucose,
        "BloodPressure":data.BloodPressure,
        "SkinThickness":data.SkinThickness,
        "Insulin":data.Insulin,
        "BMI":BMI,
        "DiabetesPedigreeFunction":data.DiabetesPedigreeFunction,
        "Age":data.Age
    }
    

def create_df(features):
    df=pd.DataFrame([features])
    return df
