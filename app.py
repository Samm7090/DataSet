'''from flask import Flask ,render_template,request
import pickle
import pandas as pd
import os

app=Flask(__name__)

base_dir=os.path.dirname(os.path.abspath(__file__))
encoder_path=os.path.join(base_dir,"label_encode.pkl")
model_path=os.path.join(base_dir,"cyber_bullies.pkl")

with open(model_path,"rb") as f:
    model=pickle.load(f)

with open(encoder_path,'rb') as f:
    le=pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route("/predict", methods=['POST'])
def predict():
    prediction = None
    try:
        tweet_text = request.form['tweet_text']
        
        # Pass input as DataFrame with column names
        input_data = pd.DataFrame([[tweet_text]],columns=["tweet_text"])
        pred_num = model.predict(input_data)[0]
        prediction= le.inverse_transform([pred_num])[0]
    except Exception as e:
        prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)'''

from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(base_dir, "label_encode.pkl")
model_path = os.path.join(base_dir, "cyber_bullies.pkl")

# Load model and encoder safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

le = None
if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
else:
    print("Warning: label_encode.pkl not found — model outputs must be strings!")

# Print label mapping for debugging
if le:
    print("\nLabel mapping (index -> label):")
    for i, cls in enumerate(le.classes_):
        print(f"{i} -> {cls}")
    print("\n")


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    try:
        tweet_text = request.form.get('tweet_text', '').strip()

        if not tweet_text:
            prediction = "Error: Empty input text"
            return render_template('index.html', prediction=prediction)

        # Prepare input data as DataFrame
        input_data = pd.DataFrame([[tweet_text]], columns=["tweet_text"])
        pred_raw = model.predict(input_data)[0]

        # Debug output to console
        print("===== DEBUG =====")
        print("Raw model output:", pred_raw, "Type:", type(pred_raw))
        if le:
            print("Encoder classes:", list(le.classes_))
        print("================")

        # Convert output to human-readable label
        if isinstance(pred_raw, str):
            prediction = pred_raw  # Model already returns string labels
        elif le:
            # If numeric, decode using label encoder
            pred_num = int(pred_raw)
            prediction = le.inverse_transform(pred_raw)[0]
        else:
            # If encoder missing, just show numeric output
            prediction = f"Predicted class index: {pred_raw}"

    except Exception as e:
        prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

