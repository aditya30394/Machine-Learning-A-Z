# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 

# There is no need to do feature scaling as the linear regression model takes 
# care of that for us

# Fitting Simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

""" Now we will visualize the results that we achieved so far """

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary VS Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
# This is the same line as that of plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.plot(X_test, y_pred, color='blue')
plt.title("Salary VS Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()