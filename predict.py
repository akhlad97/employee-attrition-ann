import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# NEW:
# hf_hub_download downloads a file from your Hugging Face repository
# to the local machine/cache when the application needs it.
from huggingface_hub import hf_hub_download



# REPO_ID = "akhlad/employee-attrition-ann" This repository contains:
#   employee_attrition_ann.keras
#   scaler.pkl
#   features_column.pkl

REPO_ID = "akhlad/employee-attrition-ann"


# Download employee_attrition_ann.keras from Hugging Face.
MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="employee_attrition_ann.keras"
)

# Download scaler.pkl from Hugging Face.

SCALER_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="scaler.pkl"
)

# Download features_column.pkl from Hugging Face 

FEATURES_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="feature_columns.pkl"
)



# Load trained model and preprocessing files,contain the LOCAL paths returned by hf_hub_download().


model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_columns = joblib.load(FEATURES_PATH)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_attrition(employee_data):
    """
    Predict whether an employee is likely to leave.

    employee_data:
        Dictionary containing employee information.

    Returns:
        prediction: "Yes" or "No"
        probability: probability of attrition
    """

    # --------------------------------------------------
    # Convert input dictionary to DataFrame
    # --------------------------------------------------

    # This line is NOT changed.
    #
    # The Flask application will send employee information as a dictionary.
    # We convert that dictionary into a DataFrame because
    # our preprocessing expects tabular data.

    input_df = pd.DataFrame([employee_data])


    # --------------------------------------------------
    # One-hot encode categorical features
    # --------------------------------------------------

    # This line is NOT changed.
    # Categorical values such as:
    #
    # BusinessTravel
    # Department
    # EducationField
    # Gender
    # JobRole
    # MaritalStatus
    # OverTime
    #
    # need to be converted into numerical columns.

    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    print("input_df after encoding:")
    print(input_df)


    # --------------------------------------------------
    # Match training feature columns
    # --------------------------------------------------

    # This line is NOT changed.
    #
    # The model was trained using a specific set of
    # encoded columns.
    #
    # features_column.pkl contains that exact column list.
    #
    # reindex() makes sure the prediction input has:
    #
    # 1. The same columns
    # 2. The same column order
    #
    # Missing columns are filled with 0.

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    print("input_df after matching training columns:")
    print(input_df)


    # --------------------------------------------------
    # Apply the same scaler used during training
    # --------------------------------------------------

    # This line is NOT changed.
    #
    # The scaler used during training must also be used
    # for new prediction data.

    # Otherwise the ANN receives data in a different
    # numerical scale than it saw during training.

    input_scaled = scaler.transform(input_df)


    # --------------------------------------------------
    # Predict probability
    # --------------------------------------------------

    # This line is NOT changed.
    #
    # The ANN returns a probability between 0 and 1.
    #
    # Example:
    #
    # 0.0191 = 1.91% probability of attrition
    #
    # 0.75 = 75% probability of attrition

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    # --------------------------------------------------
    # Convert probability to class
    # --------------------------------------------------

    # This line is NOT changed.
    #
    # We use 0.5 as the classification threshold.
    #
    # probability >= 0.5 → Yes
    # probability < 0.5  → No

    prediction = "Yes" if probability >= 0.5 else "No"


    # Return both the prediction and probability.

    return prediction, probability


# --------------------------------------------------
# Example prediction
# --------------------------------------------------

if __name__ == "__main__":

    # Example employee information.
    #
    # This section is kept from your original file.
    # It is useful for testing predict.py independently
    # before connecting Flask.

    employee = {
        "Age": 30,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Medical",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 70,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }


    # Call prediction function.

    prediction, probability = predict_attrition(employee)


    # Display result.

    print("Employee Attrition Prediction")
    print("-----------------------------")
    print("Prediction:", prediction)
    print(
        "Attrition Probability:",
        round(probability, 4)
    )
    print(
        "Attrition Probability in (%):",
        round(probability * 100, 2),
        "%"
    )