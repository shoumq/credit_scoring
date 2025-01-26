from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

y_pred = classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

predict_data = pd.DataFrame({
    'person_age': [22.0],
    'person_gender': ['female'],
    'person_education': ['Master'],
    'person_income': [71948.0],
    'person_emp_exp': [0],
    'person_home_ownership': ['RENT'],
    'loan_amnt': [35000.0],
    'loan_intent': ['PERSONAL'],
    'loan_int_rate': [16.02],
    'loan_percent_income': [0.89],
    'cb_person_cred_hist_length': [2.0],
    'credit_score': [561],
    'previous_loan_defaults_on_file': ['Yes'],
})

predict_data_encoded = pd.get_dummies(predict_data, drop_first=True)
predict_data_encoded = predict_data_encoded.reindex(columns=X_train.columns, fill_value=0)
predict_data_encoded_scaled = scaler.transform(predict_data_encoded)

new_prediction = classifier.predict(predict_data_encoded_scaled)
print(f'Prediction for new data: {new_prediction}')
