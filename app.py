# app.py
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']])
    features = scaler.transform([features])
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return jsonify({'prediction': int(prediction[0]), 'probability': probability.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
