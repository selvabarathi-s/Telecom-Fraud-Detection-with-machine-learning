import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Dataset


dataset = pd.read_csv("Card Fraud.csv")


# Input and Output


X = dataset.drop("Fraud", axis=1)
y = dataset["Fraud"]


# Split Dataset


x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Feature Scaling


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)


# Train Model


model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)


# Prediction


y_pred = model.predict(x_test)


# Accuracy


accuracy = accuracy_score(y_test, y_pred)

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy  :", model.score(x_test, y_test))

print("Accuracy Score    :", accuracy)


# Save Model


with open("model_pickle.pkl", "wb") as f:

    pickle.dump((model, scaler), f)

print("\nModel Saved Successfully...")