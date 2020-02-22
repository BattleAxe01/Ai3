# Prepare data
# pyhton Imports
import numpy as np
import matplotlib as plt
import pandas as pd

# read data
dataset = pd.read_csv("./data/bank_data.csv")
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Transform text variables into dummy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_x_gender = LabelEncoder()
x[:, 2] = le_x_gender.fit_transform(x[:, 2])
ct_x_country = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder="passthrough")
x = ct_x_country.fit_transform(x)
x = x[:, 1:]

# splitting into train and test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Neural Network
# import Keras OP
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicialize the neural network
nn = Sequential()

# Adding layers
input_size = len(x[0])  # aka 11
output_size = 1
nodes_per_layer = (input_size + output_size) // 2

nn.add(Dense(nodes_per_layer, activation="relu", input_dim=input_size))
nn.add(Dense(nodes_per_layer, activation="relu"))
nn.add(Dense(output_size, activation="sigmoid"))

nn.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
