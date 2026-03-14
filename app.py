
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_ACCURACY = 100.0
# UI improvement update

@app.route('/')
def home():
    return render_template('index.html', accuracy=MODEL_ACCURACY)

@app.route('/prediction', methods=['POST'])
def prediction():
    applicant_income = float(request.form['applicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    credit_history = float(request.form['credit_history'])

    emi = loan_amount / loan_term
    ratio = emi / applicant_income

    features = np.array([[applicant_income, loan_amount,
                          loan_term, credit_history, emi, ratio]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if prediction == 1:
        result = "Approved"
    else:
        result = "Not Approved"

    return render_template('prediction.html',
                           prediction_text=result,
                           probability=round(probability,2),
                           emi=round(emi,2),
                           ratio=round(ratio,3),
                           accuracy=MODEL_ACCURACY)

if __name__ == '__main__':
    app.run(debug=True)
