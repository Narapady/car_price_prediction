import streamlit as st
import pandas as pd
import numpy as np
import pickle


def load_model():
    with open("rf_regression.pkl", "rb") as file:
        data = pickle.load(file)
    return data["model"]


def onehot_encode_fuel(fuel_type):
    if fuel_type == 'Petrol':
        return [0, 1]
    if fuel_type == 'Diesel':
        return [1, 0]
    return [0, 0]


def onehot_encode_dealer(dealer_type):
    if dealer_type == "Dealer":
        return [0]
    return [1]


def onehot_encode_transmition(transmition_type):
    if transmition_type == "Automatic":
        return [0]
    return [1]


def show_page():
    st.title("Car Price Prediction")
    st.write("""### Fill out required information bellow for prediction""")

    fuel_type = ["Petrol", "Diesel", "CNG"]
    dealer_type = ["Dealer", "Individual"]
    transmition_type = ["Automatic", "Manual"]

    current_price = st.slider("Current Price", 1000, 140000, 1000)
    km = st.slider("Kilometers Driven", 1, 500000, 1)
    num_of_owners = st.slider("Number of owner", 0, 3, 0)
    year = st.slider("Number of Years", 1, 21, 1)
    fuel = st.selectbox("Fuel Type", fuel_type)
    dealer = st.selectbox("Dealer Type", dealer_type)
    transmition = st.selectbox("Transmition", transmition_type)

    fuel_1hot = onehot_encode_fuel(fuel)
    dealer_1hot = onehot_encode_dealer(dealer)
    transmition_1hot = onehot_encode_transmition(transmition)

    fuel_dealer = np.append(fuel_1hot, dealer_1hot)
    fuel_dealer_tran = np.append(fuel_dealer, transmition_1hot)

    button = st.button("Predict Price")
    if button:
        X = np.array([current_price, km, num_of_owners, year])
        X = np.append(X, fuel_dealer_tran)
        X = np.array([X])

        regressor = load_model()
        price = regressor.predict(X)

        st.subheader(f"The Vehicle is ${price[0]:0.2f}")
        
show_page()
