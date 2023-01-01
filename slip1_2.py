# Importing the libraries 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score 

# Loading the dataset 
data = pd.read_csv('Score.csv') 

# Separating the features and target variable 
X = data.iloc[:,0:1].values 
y = data.iloc[:,0:1].values 

# Fitting the Linear Regression model to the dataset 
regressor = LinearRegression() 
regressor.fit(X, y) 

# Predicting the results 
y_pred = regressor.predict(X) 

# Calculating the errors 
MAE = mean_absolute_error(y, y_pred) 
MSE = mean_squared_error(y, y_pred) 
RMSE = np.sqrt(MSE) 

# Printing the errors 
print("Mean Absolute Error:", MAE) 
print("Mean Squared Error:", MSE) 
print("Root Mean Squared Error:", RMSE)
print()