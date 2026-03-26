# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# ---------------- Load Model ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "medical_charges_model.joblib")
model = joblib.load(MODEL_PATH)

# ---------------- FastAPI App ----------------
app = FastAPI(title="Medical Insurance Prediction API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Input Schema ----------------
class InputData(BaseModel):
    age: float
    bmi: float
    children: int
    sex: str        # male/female
    smoker: str     # yes/no

# ---------------- Encoding Helpers ----------------
def encode_sex(sex: str) -> int:
    if sex.lower() == "male":
        return 1
    elif sex.lower() == "female":
        return 0
    else:
        raise ValueError("sex must be 'male' or 'female'")

def encode_smoker(smoker: str) -> int:
    if smoker.lower() == "yes":
        return 1
    elif smoker.lower() == "no":
        return 0
    else:
        raise ValueError("smoker must be 'yes' or 'no'")

# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
def predict(data: InputData):
    try:
        features = [[
            data.age,
            data.bmi,
            data.children,
            encode_smoker(data.smoker),
            encode_sex(data.sex)
        ]]
        prediction = model.predict(features)[0]
        return {"predicted_charges": round(float(prediction), 2)}
    except Exception as e:
        return {"error": str(e)}
