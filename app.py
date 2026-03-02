from fastapi import FastAPI
import joblib

app = FastAPI()

# load model
model = joblib.load("models/job_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.get("/")
def home():
    return {"message": "Job ML API Running 🚀"}

@app.get("/predict")
def predict(skill: str):
    vec = vectorizer.transform([skill])
    result = model.predict(vec)
    return {"prediction": result[0]}