# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:20:19 2018

@author: Aditya Kumar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# no need to split the data into test and train as data is very less

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg_object = LinearRegression()
poly_reg_object.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X, linear_reg.predict(X), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, poly_reg_object.predict( poly_reg.fit_transform(X)), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# To get a continuous graph
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, poly_reg_object.predict( poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Final result comparision

linear_reg.predict(6.5)
poly_reg_object.predict(poly_reg.fit_transform(6.5))