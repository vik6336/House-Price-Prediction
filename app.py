import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Load your data and model training code (or load a pre-trained model)
@st.cache_data
def load_data():
    data = pd.read_csv('AmesHousing.csv')
    data = data.dropna(subset=['SalePrice'])
    X = data.drop('SalePrice', axis=1)
    # Drop identifier and non-predictive columns
    cols_to_drop = ['Order', 'PID', 'Mo Sold', 'Yr Sold']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    y = data['SalePrice']
    X_numeric = pd.get_dummies(X)
    X_numeric = X_numeric.dropna()
    y_aligned = y.loc[X_numeric.index]
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_numeric)
    return X, X_numeric, X_norm, y_aligned, scaler

X, X_numeric, X_norm, y_aligned, scaler = load_data()

# Train model (or load a pre-trained model)
@st.cache_resource
def train_model(X_norm, y_aligned):
    sgdr = SGDRegressor(max_iter=1000, learning_rate='adaptive', eta0=0.01)
    sgdr.fit(X_norm, y_aligned)
    return sgdr

sgdr = train_model(X_norm, y_aligned)

st.title("Ames Housing Price Prediction")

# User input for features
st.header("Input Features")
input_dict = {}
for col in X.columns[:10]:  # Show first 10 features for demo; expand as needed
    if pd.api.types.is_numeric_dtype(X[col]):
        input_dict[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    else:
        input_dict[col] = st.selectbox(f"{col}", X[col].unique())

# Fill missing features with mean/mode from training data
for col in X.columns:
    if col not in input_dict:
        if pd.api.types.is_numeric_dtype(X[col]):
            input_dict[col] = X[col].mean()
        else:
            input_dict[col] = X[col].mode()[0]

# Convert user input to DataFrame and preprocess
input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X_numeric.columns, fill_value=0)
input_norm = scaler.transform(input_df)

# Make prediction
prediction = sgdr.predict(input_norm)[0]
prediction = max(0, prediction)  # Ensure prediction is not negative
st.subheader(f"Predicted Sale Price: ${prediction:,.0f}")

# SHAP explanation
st.header("Model Explanation (SHAP)")
shap_sample = X_numeric.sample(n=100, random_state=42).astype(float)
explainer = shap.Explainer(sgdr, shap_sample)
shap_values = explainer(shap_sample)

# SHAP summary plot
fig = plt.figure()
shap.summary_plot(shap_values, shap_sample, show=False)
st.pyplot(fig)
