from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained pipeline
with open('pipe.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'age': float(request.form['age']),
            'heart_rate': float(request.form['heart_rate']),
            'systolic_blood_pressure': float(request.form['systolic_blood_pressure']),
            'oxygen_saturation': float(request.form['oxygen_saturation']),
            'body_temperature': float(request.form['body_temperature']),
            'pain_level': int(request.form['pain_level']),
            'chronic_disease_count': int(request.form['chronic_disease_count']),
            'previous_er_visits': int(request.form['previous_er_visits']),
            'arrival_mode': request.form['arrival_mode']
        }

        # Convert to DataFrame (required for the pipeline)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df).max() * 100

        # Triage level meaning
        triage_levels = {
            0: "Level 0 - Non-Urgent",
            1: "Level 1 - Less Urgent",
            2: "Level 2 - Urgent",
            3: "Level 3 - Emergent / Life-threatening"
        }

        result = triage_levels.get(prediction, "Unknown")

        return render_template('index.html',
                               prediction_text=result,
                               probability=round(probability, 2),
                               input_data=data)

    except Exception as e:
        return render_template('index.html', error=str(e))


# Production start command
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)