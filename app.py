from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('liver_model.pkl', 'rb'))
scaler = pickle.load(open('liver_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data = np.array(data).reshape(1, -1)

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "⚠️ Liver Disease Detected"
        else:
            result = "✅ Healthy Liver"

        return render_template('index.html', prediction_text=result)

    except:
        return render_template('index.html', prediction_text="⚠️ Invalid Input!")

if __name__ == "__main__":
    app.run(debug=True)