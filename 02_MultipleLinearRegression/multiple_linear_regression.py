# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# The dataset has a categorical feature (State)
# This needs to be label encoded
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()

# This will contain all the dummy variable,
# but its always a good practice to avoid one dummy variable from the total
# So take away feature at index : 0
X = X[:,1:]
