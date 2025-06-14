from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load("models/model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}

    if request.method == "POST":
        # Collect form data
        form_data['age'] = request.form['age']
        form_data['sex'] = request.form['sex']
        form_data['cp'] = request.form['cp']
        form_data['bp'] = request.form['bp']
        form_data['chol'] = request.form['chol']
        form_data['fbs'] = request.form['fbs']
        form_data['restecg'] = request.form['restecg']
        form_data['thalach'] = request.form['thalach']
        form_data['exang'] = request.form['exang']
        form_data['oldpeak'] = request.form['oldpeak']
        form_data['slope'] = request.form['slope']

        # Now convert form_data values into input format for prediction
        # You need to map these form values into numerical input just like you did before

        # Example only (you have to modify as per your encoding logic):
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

        input_data = [
            int(form_data['age']),
            sex_map[form_data['sex']],
            cp_map[form_data['cp']],
            int(form_data['bp']),
            int(form_data['chol']),
            fbs_map[form_data['fbs']],
            restecg_map[form_data['restecg']],
            int(form_data['thalach']),
            exang_map[form_data['exang']],
            float(form_data['oldpeak']),
            slope_map[form_data['slope']]
        ]

        prediction = model.predict([input_data])[0]

        if prediction == 1:
            result = "⚠️ Patient is likely to have Heart Disease"
        else:
            result = "✅ No Heart Disease detected"

    return render_template("index.html", result=result, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
