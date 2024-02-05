#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]
abalone_data = pd.read_csv(url, header=None, names=column_names)

# Convert categorical variable "Sex" to numerical using one-hot encoding
abalone_data = pd.get_dummies(abalone_data, columns=["Sex"], drop_first=True)

# Features and target variable
X = abalone_data.drop("Rings", axis=1)
y = abalone_data["Rings"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

# Ridge Regression
ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

# Bagging Regression
bagging_model = BaggingRegressor(n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# Evaluate models
def evaluate_model(name, predictions, y_true):
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    print(f"{name} RMSE: {rmse}")

# Print the evaluation results
evaluate_model("Linear Regression", linear_pred, y_test)
evaluate_model("Lasso Regression", lasso_pred, y_test)
evaluate_model("Ridge Regression", ridge_pred, y_test)
evaluate_model("Bagging Regression", bagging_pred, y_test)
evaluate_model("Random Forest", rf_pred, y_test)
evaluate_model("LightGBM", lgb_pred, y_test)


# In[ ]:




