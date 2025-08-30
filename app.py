from flask import Flask, render_template, request
import pickle
import pandas as pd   
import os

app = Flask(__name__)
	# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
	# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route("/predict", methods=['POST'])
def predict():
    prediction = None
    try:
        age = float(request.form['Age'])
        sex = request.form['Sex']
        bp = request.form['BP']
        chol = request.form['Cholesterol']
        na_to_k = float(request.form['Na_to_K'])
        # Pass input as DataFrame with column names
        input_data = pd.DataFrame([[age, sex, bp, chol, na_to_k]], 
	                                  columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"])
        prediction = model.predict(input_data)[0]
    except Exception as e:
        prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
		
