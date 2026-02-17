from fastapi import FastAPI
import joblib

app = FastAPI(title="Predictive Analytics System")

model = joblib.load("model/model.pkl")


@app.get("/")
def root():
    return {"message": "Business Forecast API"}


@app.post("/predict")
def predict(marketing: float, price: float, month: int):
    result = model.predict([[marketing, price, month]])
    return {"predicted_revenue": float(result[0])}
