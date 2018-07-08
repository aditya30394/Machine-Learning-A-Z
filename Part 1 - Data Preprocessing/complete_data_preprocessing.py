# Data Preprocessing

""" import important libraries """
# numpy is the library we use for doing mathematics
import numpy as np
# matplotlib.pyplot library is used for data visualization
import matplotlib.pyplot as plt
# pandas is the library we use for importing and exporting the datasets
import pandas as pd

""" importing the dataset from csv file """
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

""" Taking care of missing values using sklearn library """
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3]);
X[:, 1:3] = imputer.transform(X[:, 1:3]);

""" Taking care of categorical data """
""" 
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])

Simply using encoder automatically puts a bias. As we Can see that Germany,
France and Spain have been given values 0 1 and 2 which might suggest to our 
ML algorithms that somehow there is order between these values. This is 
obviously wrong and to correct this we would use a concept of dummy variables 
where this one column would be splitup into multiple columns 
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Independent Variable
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])
onehotencode = OneHotEncoder(categorical_features = [0])
X = onehotencode.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
""" 
As we are doing the classification we don't need to do feature scaling on 
the Dependent variable. However for regression problem we might need to do
the feature scaling for dependent variable too
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
"""