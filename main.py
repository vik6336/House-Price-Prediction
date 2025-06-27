import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
np.set_printoptions(precision=2)

# Load data
data = pd.read_csv('AmesHousing.csv')

# Drop rows with missing target
data = data.dropna(subset=['SalePrice'])

# Separate features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# One-hot encode categorical features
X_numeric = pd.get_dummies(X)

# Drop rows with any remaining missing values in features
X_numeric = X_numeric.dropna()
# Align y with X_numeric (in case rows were dropped)
y_aligned = y.loc[X_numeric.index]

# Standardize features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_numeric)

# Convert boolean columns to int (0/1) for compatibility with np.ptp
X_numeric_int = X_numeric.astype(int)

# Print peak-to-peak range
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_numeric_int, axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm, axis=0)}")

# Fit model
sgdr = SGDRegressor(max_iter=1000, learning_rate='adaptive', eta0=0.01)
sgdr.fit(X_norm, y_aligned)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
assert w_norm is not None and b_norm is not None
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:10]}" )
print(f"Target values \n{y_aligned[:10]}")

# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,10,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_norm[:,i],y_aligned, label = 'target')
    ax[i].set_xlabel(X_numeric.columns[i])
    ax[i].scatter(X_norm[:,i],y_pred,color="orange", label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
ax[0].get_yaxis().set_major_formatter(FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# SHAP analysis for model interpretability
explainer = shap.Explainer(sgdr, X_numeric)
shap_values = explainer(X_numeric)
shap.summary_plot(shap_values, X_numeric, show=True)