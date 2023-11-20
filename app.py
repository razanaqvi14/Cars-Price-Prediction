from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("random_forest_regression_model.pkl", "rb"))


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        present_price = float(request.form["present_price"])

        kms_driven = float(request.form["kms_driven"])

        owner = int(request.form["owner"])

        year = int(request.form["year"])
        year = 2023 - year

        fuel_type = request.form["fuel_type"]
        fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

        seller_type = request.form["seller_type"]
        seller_type_individual = 1 if seller_type == "Individual" else 0

        transmission_type = request.form["transmission_type"]
        transmission_manual = 1 if transmission_type == "Manual" else 0

        prediction_data = np.array(
            [
                [
                    present_price,
                    kms_driven,
                    owner,
                    year,
                    fuel_type_diesel,
                    fuel_type_petrol,
                    seller_type_individual,
                    transmission_manual,
                ]
            ]
        )

        predicted_price = model.predict(prediction_data)[0]

        return render_template(
            "index.html",
            prediction_text=f"You can sell this car for {predicted_price:.2f} Lacs",
        )


if __name__ == "__main__":
    app.run(debug=True)
