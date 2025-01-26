from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = FastAPI()

class LoanData(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str

df = pd.read_csv('loan_data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression(solver='liblinear', max_iter=200)
classifier.fit(X_train_scaled, y_train)

def metrics():
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {"accuracy": accuracy, "report": report}

@app.post("/predict")
def predict(data: LoanData):
    predict_data = pd.DataFrame([{
        'person_age': data.person_age,
        'person_gender': data.person_gender,
        'person_education': data.person_education,
        'person_income': data.person_income,
        'person_emp_exp': data.person_emp_exp,
        'person_home_ownership': data.person_home_ownership,
        'loan_amnt': data.loan_amnt,
        'loan_intent': data.loan_intent,
        'loan_int_rate': data.loan_int_rate,
        'loan_percent_income': data.loan_percent_income,
        'cb_person_cred_hist_length': data.cb_person_cred_hist_length,
        'credit_score': data.credit_score,
        'previous_loan_defaults_on_file': data.previous_loan_defaults_on_file,
    }])

    predict_data_encoded = pd.get_dummies(predict_data, drop_first=True)
    predict_data_encoded = predict_data_encoded.reindex(columns=X_train.columns, fill_value=0)
    predict_data_encoded_scaled = scaler.transform(predict_data_encoded)

    new_prediction = classifier.predict(predict_data_encoded_scaled)

    prediction_result = int(new_prediction[0])

    return {"prediction": prediction_result}
