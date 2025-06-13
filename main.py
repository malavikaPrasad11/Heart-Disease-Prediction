import joblib
import pandas as pd
import os

def load_model(model_path="models/model.pkl"):
    return joblib.load(model_path)

def predict(model, input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return "Heart Disease" if prediction == 1 else "Normal"

if __name__ == "__main__":
    # Sample input (you can replace with dynamic input later)
    sample_input = {
        "age": 63,
        "sex": 1,
        "chest pain type": 3,
        "resting bp s": 145,
        "cholesterol": 233,
        "fasting blood sugar": 1,
        "resting ecg": 0,
        "max heart rate": 150,
        "exercise angina": 0,
        "oldpeak": 2.3,
        "ST slope": 0
    }

    model = load_model()
    result = predict(model, sample_input)
    print("Prediction:", result)
