from flask import Flask, render_template, request
import pandas as pd
import joblib

import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "churn_model.pkl")
columns_path = os.path.join(BASE_DIR, "churn_columns.pkl")

model = joblib.load(model_path)
columns = joblib.load(columns_path)

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None

    if request.method == "POST":

        # === Récupération données ===
        form = request.form

        data = {}

        # Variables numériques
        data["SeniorCitizen"] = int(form["SeniorCitizen"])
        data["tenure"] = float(form["tenure"])
        data["MonthlyCharges"] = float(form["MonthlyCharges"])
        data["TotalCharges"] = float(form["TotalCharges"])

        # === One-Hot Encoding ===

        data["gender_Male"] = 1 if form["gender"] == "Male" else 0
        data["Partner_Yes"] = 1 if form["Partner"] == "Yes" else 0
        data["Dependents_Yes"] = 1 if form["Dependents"] == "Yes" else 0
        data["PhoneService_Yes"] = 1 if form["PhoneService"] == "Yes" else 0

        data["MultipleLines_Yes"] = 1 if form["MultipleLines"] == "Yes" else 0
        data["MultipleLines_No phone service"] = 1 if form["MultipleLines"] == "No phone service" else 0

        data["InternetService_Fiber optic"] = 1 if form["InternetService"] == "Fiber optic" else 0
        data["InternetService_No"] = 1 if form["InternetService"] == "No" else 0

        data["OnlineSecurity_Yes"] = 1 if form["OnlineSecurity"] == "Yes" else 0
        data["OnlineBackup_Yes"] = 1 if form["OnlineBackup"] == "Yes" else 0
        data["DeviceProtection_Yes"] = 1 if form["DeviceProtection"] == "Yes" else 0
        data["TechSupport_Yes"] = 1 if form["TechSupport"] == "Yes" else 0
        data["StreamingTV_Yes"] = 1 if form["StreamingTV"] == "Yes" else 0
        data["StreamingMovies_Yes"] = 1 if form["StreamingMovies"] == "Yes" else 0

        data["Contract_One year"] = 1 if form["Contract"] == "One year" else 0
        data["Contract_Two year"] = 1 if form["Contract"] == "Two year" else 0

        data["PaperlessBilling_Yes"] = 1 if form["PaperlessBilling"] == "Yes" else 0

        data["PaymentMethod_Credit card (automatic)"] = 1 if form["PaymentMethod"] == "Credit card (automatic)" else 0
        data["PaymentMethod_Electronic check"] = 1 if form["PaymentMethod"] == "Electronic check" else 0
        data["PaymentMethod_Mailed check"] = 1 if form["PaymentMethod"] == "Mailed check" else 0

        # === Alignement colonnes ===
        df = pd.DataFrame([data])
        df = df.reindex(columns=columns, fill_value=0)

        # === Prediction ===
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        result = "Client va churn ❌" if pred == 1 else "Client fidèle ✅"
        probability = round(prob * 100, 2)

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)