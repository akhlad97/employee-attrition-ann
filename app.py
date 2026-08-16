from flask import Flask, render_template, request

# --------------------------------------------------
# OLD CODE
# --------------------------------------------------
# Previously, if prediction logic was written directly
# inside app.py, we may have loaded the model here.
#
# Example OLD code:
#
# from tensorflow.keras.models import load_model
# import joblib
#
# model = load_model("employee_attrition_ann.keras")
# scaler = joblib.load("scaler.pkl")
# feature_columns = joblib.load("feature_columns.pkl")
#
# --------------------------------------------------
# WHY COMMENTED?
# --------------------------------------------------
# We don't need to load these files inside app.py anymore.
#
# Our predict.py now downloads:
#   employee_attrition_ann.keras
#   scaler.pkl
#   feature_columns.pkl
#
# directly from Hugging Face.
#
# Therefore app.py only needs to call predict_attrition()
# from predict.py.


# --------------------------------------------------
# NEW CODE: Import prediction function
# --------------------------------------------------

from predict import predict_attrition

# This connects Flask to our prediction.py file.
#
# predict.py is responsible for:
#   1. Downloading model from Hugging Face
#   2. Downloading scaler from Hugging Face
#   3. Downloading feature_columns from Hugging Face
#   4. Preprocessing the employee data
#   5. Making the ANN prediction


# --------------------------------------------------
# Create Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# Prediction Route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------
    # Collect employee information from HTML form
    # --------------------------------------------------

    employee = {

        "Age": int(request.form["Age"]),

        "BusinessTravel": request.form["BusinessTravel"],

        "DailyRate": int(request.form["DailyRate"]),

        "Department": request.form["Department"],

        "DistanceFromHome": int(request.form["DistanceFromHome"]),

        "Education": int(request.form["Education"]),

        "EducationField": request.form["EducationField"],

        "EnvironmentSatisfaction":
            int(request.form["EnvironmentSatisfaction"]),

        "Gender": request.form["Gender"],

        "HourlyRate": int(request.form["HourlyRate"]),

        "JobInvolvement":
            int(request.form["JobInvolvement"]),

        "JobLevel":
            int(request.form["JobLevel"]),

        "JobRole":
            request.form["JobRole"],

        "JobSatisfaction":
            int(request.form["JobSatisfaction"]),

        "MaritalStatus":
            request.form["MaritalStatus"],

        "MonthlyIncome":
            int(request.form["MonthlyIncome"]),

        "MonthlyRate":
            int(request.form["MonthlyRate"]),

        "NumCompaniesWorked":
            int(request.form["NumCompaniesWorked"]),

        "OverTime":
            request.form["OverTime"],

        "PercentSalaryHike":
            int(request.form["PercentSalaryHike"]),

        "PerformanceRating":
            int(request.form["PerformanceRating"]),

        "RelationshipSatisfaction":
            int(request.form["RelationshipSatisfaction"]),

        "StockOptionLevel":
            int(request.form["StockOptionLevel"]),

        "TotalWorkingYears":
            int(request.form["TotalWorkingYears"]),

        "TrainingTimesLastYear":
            int(request.form["TrainingTimesLastYear"]),

        "WorkLifeBalance":
            int(request.form["WorkLifeBalance"]),

        "YearsAtCompany":
            int(request.form["YearsAtCompany"]),

        "YearsInCurrentRole":
            int(request.form["YearsInCurrentRole"]),

        "YearsSinceLastPromotion":
            int(request.form["YearsSinceLastPromotion"]),

        "YearsWithCurrManager":
            int(request.form["YearsWithCurrManager"])
    }


    # --------------------------------------------------
    # OLD APPROACH
    # --------------------------------------------------

    # Previously, prediction might have been performed
    # directly using the model loaded inside app.py.
    #
    # Example:
    #
    # prediction_probability = model.predict(input_data)
    #
    # This is commented because prediction logic has
    # been moved to predict.py.
    #
    # Keeping prediction logic in predict.py makes the
    # application cleaner and easier to deploy.


    # --------------------------------------------------
    # NEW APPROACH
    # --------------------------------------------------

    prediction, probability = predict_attrition(employee)

    # predict_attrition() comes from predict.py.
    #
    # It handles:
    #
    # employee data
    #       ↓
    # pandas DataFrame
    #       ↓
    # one-hot encoding
    #       ↓
    # feature column matching
    #       ↓
    # scaling
    #       ↓
    # Hugging Face ANN model
    #       ↓
    # prediction + probability


    # --------------------------------------------------
    # Convert probability into percentage
    # --------------------------------------------------

    probability_percent = round(probability * 100, 2)


    # --------------------------------------------------
    # Send result to HTML
    # --------------------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability_percent
    )


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # OLD CODE
    # --------------------------------------------------

    # app.run(debug=True)
    #
    # This was suitable for local development.
    #
    # We are not using it for the final deployment because
    # deployment platforms such as Render/AWS provide a PORT
    # through an environment variable.


    # --------------------------------------------------
    # NEW CODE
    # --------------------------------------------------

    import os

    # Get PORT from the deployment environment.
    #
    # If PORT is not provided, use 5000 for local testing.

    port = int(os.environ.get("PORT", 5000))


    # host="0.0.0.0" allows Flask to accept connections
    # from outside the local machine/container.
    #
    # This is important when we later run the application
    # inside Docker/ECS/Fargate.

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )