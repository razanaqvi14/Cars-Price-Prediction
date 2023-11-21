from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("random_forest_regression_model.pkl", "rb"))

years = list(range(1993, 2024))


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html", years=years)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        year = 2023 - int(request.form["year"])

        present_price = float(request.form["present_price"])

        kilometers_driven = float(request.form["kilometers_driven"])

        owner = int(request.form["owner"])

        fuel_type = request.form["fuel_type"]
        fuel_type_diesel = 1 if fuel_type == "diesel" else 0
        fuel_type_petrol = 1 if fuel_type == "petrol" else 0

        seller_type = request.form["seller_type"]
        seller_type_individual = 1 if seller_type == "individual" else 0

        transmission_type = request.form["transmission_type"]
        transmission_manual = 1 if transmission_type == "manual" else 0

        predicted_price = round(
            model.predict(
                np.array(
                    [
                        [
                            present_price,
                            kilometers_driven,
                            owner,
                            year,
                            fuel_type_diesel,
                            fuel_type_petrol,
                            seller_type_individual,
                            transmission_manual,
                        ]
                    ]
                )
            )[0],
            2,
        )

        return render_template(
            "index.html",
            years=years,
            prediction_text=f"You can sell this car for {predicted_price:.2f} Lacs",
        )


if __name__ == "__main__":
    app.run(debug=True)
