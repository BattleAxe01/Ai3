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

# Feature Scalling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Neural Network
# import Keras OP
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Inicialize the neural network
def build_classifier():
    classifier = Sequential()

    # Adding layers
    input_size = len(x[0])  # aka 11
    output_size = 1
    nodes_per_layer = (input_size + output_size) // 2
    drop = 0.1

    classifier.add(Dense(nodes_per_layer, activation="relu", input_dim=input_size))
    classifier.add(Dropout(rate=drop))
    classifier.add(Dense(nodes_per_layer, activation="relu"))
    classifier.add(Dropout(rate=drop))
    classifier.add(Dense(output_size, activation="sigmoid"))

    # Compile classifier
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


nn = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10)

# Train the neural network
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=nn, X=x_train, y=y_train, cv=10, n_jobs=-1)

print("Media:", accuracy.mean())
print("Variancia:", accuracy.std())

# Make Prediction
# y_pred = nn.predict(x_test)
# y_pred = (y_pred > 0.5)
#
# # Comparing results
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(y_test, y_pred)
#
# print("Acurracy on test:", (cm[0][0] + cm[1][1]) / len(y_test))
