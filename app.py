import streamlit as st
import pandas as pd
import pickle

# Load Model

with open("model_pickle.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Title

st.title("Telecom Fraud Detection")
st.write("Enter the details below to predict whether the transaction is Fraud or Genuine.")

st.write("--------------------------------")

# User Inputs

distance1 = st.number_input("Distance From Home", min_value=0.0)

distance2 = st.number_input("Distance From Last Transaction", min_value=0.0)

ratio = st.number_input("Purchase Ratio", min_value=0.0)

repeat = st.selectbox(
    "Repeat Retailer",
    [0, 1]
)

card = st.selectbox(
    "Card Used",
    [0, 1]
)

pin = st.selectbox(
    "PIN Used",
    [0, 1]
)

online = st.selectbox(
    "Online Transaction",
    [0, 1]
)

st.write("--------------------------------")

# Prediction

if st.button("Predict"):

    data = pd.DataFrame({

        "Distance1":[distance1],
        "Distance2":[distance2],
        "Ratio":[ratio],
        "Repeat":[repeat],
        "Card":[card],
        "Pin":[pin],
        "Online":[online]

    })

    data = scaler.transform(data)

    result = model.predict(data)

    probability = model.predict_proba(data)

    if result[0] == 1:

        st.error("Fraud Transaction Detected")

    else:

        st.success("Genuine Transaction")

    st.write("--------------------------------")

    st.subheader("Prediction Probability")

    st.write("Genuine :", round(probability[0][0] * 100, 2), "%")

    st.write("Fraud :", round(probability[0][1] * 100, 2), "%")