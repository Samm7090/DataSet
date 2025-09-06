from flask import Flask, render_template, request
import pickle
import pandas as pd   
import os
import numpy as np

app=Flask(__name__)

base_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_dir,'De_tree_model.pkl')
encoder_path=os.path.join(base_dir,'label_encoder.pkl')

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    le = pickle.load(f)

df=pd.read_csv('fetal_health.csv')
feature_names=df.drop(columns=['fetal_health']).columns.tolist()

@app.route("/")
def home():
    return render_template("index.html",feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values (convert to float)
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])   # shape (1, n_features)

        # Predict
        prediction = model.predict(final_features)       # numeric
        decoded_pred = le.inverse_transform(prediction)  # decoded label

        return render_template(
            "index.html",
            feature_names=feature_names,
            prediction_text=f"Fetal health is: {decoded_pred[0]}"
        )

    except Exception as e:
        return render_template("index.html",feature_names=feature_names, prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)