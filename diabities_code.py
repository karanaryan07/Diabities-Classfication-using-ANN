# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 03:16:21 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot 

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[: ,0:8].values
y = dataset.iloc[: , 8].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, 0])
X[:, 0] = imputer.transform(X[:, 0])


from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.24 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 5 , init = 'uniform' , activation = 'relu' , input_dim = 8))

classifier.add(Dense(output_dim = 5 , init = 'uniform' , activation = 'relu'))

classifier.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid'))


classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 2, epochs = 200)

'''
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
