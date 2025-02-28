from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8001/predict"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error_message = None

    if request.method == "POST":
        # Récupérer les valeurs du formulaire
        input_data = {
            "State": request.form["State"],
            "Account_length": int(request.form["Account_length"]),
            "Area_code": int(request.form["Area_code"]),
            "International_plan": request.form["International_plan"],
            "Voice_mail_plan": request.form["Voice_mail_plan"],
            "Number_vmail_messages": int(request.form["Number_vmail_messages"]),
            "Total_day_minutes": float(request.form["Total_day_minutes"]),
            "Total_day_calls": int(request.form["Total_day_calls"]),
            "Total_day_charge": float(request.form["Total_day_charge"]),
            "Total_eve_minutes": float(request.form["Total_eve_minutes"]),
            "Total_eve_calls": int(request.form["Total_eve_calls"]),
            "Total_eve_charge": float(request.form["Total_eve_charge"]),
            "Total_night_minutes": float(request.form["Total_night_minutes"]),
            "Total_night_calls": int(request.form["Total_night_calls"]),
            "Total_night_charge": float(request.form["Total_night_charge"]),
            "Total_intl_minutes": float(request.form["Total_intl_minutes"]),
            "Total_intl_calls": int(request.form["Total_intl_calls"]),
            "Total_intl_charge": float(request.form["Total_intl_charge"]),
            "Customer_service_calls": int(request.form["Customer_service_calls"])
        }

        # Envoyer la requête à l'API FastAPI
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            prediction = response.json().get("prediction")
        else:
            error_message = f"Erreur : {response.json()}"

    return render_template("index.html", prediction=prediction, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

