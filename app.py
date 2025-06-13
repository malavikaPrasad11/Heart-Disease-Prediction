from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load(os.path.join("models", "model.pkl"))

# Mapping dictionaries (same as Streamlit)
sex_map = {"Male": 1, "Female": 0}
cp_map = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4
}
fbs_map = {"< 120 mg/dL": 0, "> 120 mg/dL": 1}
restecg_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        data = {
            "age": int(request.form["age"]),
            "sex": sex_map[request.form["sex"]],
            "chest pain type": cp_map[request.form["cp"]],
            "resting bp s": int(request.form["bp"]),
            "cholesterol": int(request.form["chol"]),
            "fasting blood sugar": fbs_map[request.form["fbs"]],
            "resting ecg": restecg_map[request.form["restecg"]],
            "max heart rate": int(request.form["thalach"]),
            "exercise angina": exang_map[request.form["exang"]],
            "oldpeak": float(request.form["oldpeak"]),
            "ST slope": slope_map[request.form["slope"]]
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
