# Ames Housing Price Prediction App

This project is an interactive Streamlit web app that predicts house prices using the Ames Housing dataset and linear regression. It demonstrates a complete machine learning workflow, from data preprocessing to model training, prediction, and interpretability with SHAP.

## Project Overview
- **Goal:** Predict the sale price of houses based on dozens of features from the Ames Housing dataset.
- **Interface:** User-friendly web app built with Streamlit.
- **Model:** Linear Regression using SGDRegressor (scikit-learn).

## Features
- Removes irrelevant features (identifiers, dates) automatically
- Handles missing values and categorical variables
- One-hot encoding for categorical features
- Feature scaling (z-score normalization)
- Interactive user input for predictions
- SHAP summary plot for model interpretability

## Data Preprocessing
1. **Load Data:** Reads `AmesHousing.csv` into a pandas DataFrame.
2. **Remove Irrelevant Features:** Drops columns like `Order`, `PID`, `Mo Sold`, `Yr Sold`.
3. **Handle Missing Values:** Drops rows with missing target and any remaining missing feature values.
4. **Feature Engineering:**
    - Separates features (`X`) and target (`y`)
    - Applies one-hot encoding to categorical features
    - Aligns target with cleaned features
5. **Feature Scaling:** Uses `StandardScaler` to normalize all features.

## Model
- **Algorithm:** Linear Regression (`SGDRegressor`)
- **Learning Rate:** Adaptive, initial value 0.01
- **Max Iterations:** 1000

## Usage Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd House-Price-Prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy scikit-learn shap matplotlib
   ```
3. **Add the dataset:**
   - Download `AmesHousing.csv` and place it in the project directory.
4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```



## What I Learned
- How to preprocess real-world data for ML
- How to build and deploy an interactive ML app with Streamlit
- How to interpret model predictions using SHAP
- The importance of removing irrelevant features for better model performance

---


