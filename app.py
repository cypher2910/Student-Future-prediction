from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)


# ----------------------------------
# Load trained model and scalers
# ----------------------------------

model = load_model("student_rnn.keras")

feature_scaler = joblib.load(
    "feature_scaler.pkl"
)

target_scaler = joblib.load(
    "target_scaler.pkl"
)


# ----------------------------------
# Home page
# ----------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            # Month 1
            month1 = [
                float(request.form["attendance1"]),
                float(request.form["marks1"]),
                float(request.form["assignment1"]),
                float(request.form["learning1"])
            ]

            # Month 2
            month2 = [
                float(request.form["attendance2"]),
                float(request.form["marks2"]),
                float(request.form["assignment2"]),
                float(request.form["learning2"])
            ]

            # Month 3
            month3 = [
                float(request.form["attendance3"]),
                float(request.form["marks3"]),
                float(request.form["assignment3"]),
                float(request.form["learning3"])
            ]


            # Combine three months
            student_data = np.array([
                month1,
                month2,
                month3
            ])


            # ----------------------------------
            # Scale features
            # ----------------------------------

            student_scaled = feature_scaler.transform(
                student_data
            )


            # RNN expects:
            # (samples, timesteps, features)
            #
            # (1 student, 3 months, 4 features)

            student_scaled = student_scaled.reshape(
                1, 3, 4
            )


            # ----------------------------------
            # Make prediction
            # ----------------------------------

            predicted_scaled = model.predict(
                student_scaled,
                verbose=0
            )


            # Convert back to percentage
            predicted = target_scaler.inverse_transform(
                predicted_scaled
            )


            prediction = round(
                float(predicted[0][0]),
                2
            )


        except Exception as e:

            error = str(e)


    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


# ----------------------------------
# Start Flask server
# ----------------------------------

if __name__ == "__main__":
    app.run(debug=True)