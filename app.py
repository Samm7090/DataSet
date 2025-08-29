from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "loan_approval.csv")  # optional

# ---------------------------
# Load model (must be the *pipeline* you saved)
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------
# (Optional) Load data to populate dropdowns
# ---------------------------
df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

# The raw feature names the pipeline expects (same names you used in training)
NUM_COLS = ["Age", "Log10_Income_USD", "CreditScore", "ExistingLoansCount", "Log10_LoanAmount_INR"]
CAT_COLS = ["EmploymentType", "MaritalStatus", "City"]
FEATURES = NUM_COLS + CAT_COLS

# Build dropdown options (fallbacks if CSV isn't present)
employment_options = sorted(df["EmploymentType"].dropna().unique().tolist()) if df is not None else ["Salaried", "Self-Employed"]
marital_options    = sorted(df["MaritalStatus"].dropna().unique().tolist()) if df is not None else ["Single", "Married"]
city_options       = sorted(df["City"].dropna().unique().tolist())          if df is not None else ["Mumbai","Delhi","Bengaluru","Chennai","Pune","Hyderabad"]

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        employment_options=employment_options,
        marital_options=marital_options,
        city_options=city_options,
        result=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Read form fields
    try:
        age = float(request.form.get("Age", 0))
        income = float(request.form.get("Log10_Income_USD", 0))
        credit = float(request.form.get("CreditScore", 0))
        loans = float(request.form.get("ExistingLoansCount", 0))
        loan_amt = float(request.form.get("Log10_LoanAmount_INR", 0))

        employment = request.form.get("EmploymentType", "Salaried")
        marital    = request.form.get("MaritalStatus", "Single")
        city       = request.form.get("City", "Mumbai")

        # Create a single-row DataFrame with EXACT training feature names
        row = {
            "Age": age,
            "Log10_Income_USD": income,
            "CreditScore": credit,
            "ExistingLoansCount": loans,
            "Log10_LoanAmount_INR": loan_amt,
            "EmploymentType": employment,
            "MaritalStatus": marital,
            "City": city
        }
        X = pd.DataFrame([row], columns=FEATURES)

        # Predict using the pipeline (handles scaling + OHE internally)
        proba = model.predict_proba(X)[:, 1][0]
        pred  = int(model.predict(X)[0])

        result = {
            "probability": round(float(proba), 3),
            "class": pred
        }
        msg = f"Approval Probability: {result['probability']:.3f} — Predicted Class: {result['class']}"
    except Exception as e:
        msg = f"Error: {e}"

    return render_template(
        "index.html",
        employment_options=employment_options,
        marital_options=marital_options,
        city_options=city_options,
        result=msg
    )

if __name__ == "__main__":
    # Run the dev server
    app.run(debug=True)
