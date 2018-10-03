# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:22:38 2018

@author: avaca
"""

"""
This script will be used for building our models and designing all the functions needed 
for improving and cross validating them
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from housing_cleaning import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
import keras
from keras import optimizers



train = pd.read_csv("train_gar.csv", sep = ",") 

where_are_infinites(train)
#con esta funcion vemos que tenemos un infinito en la fila 560 de la columna Bedroom_TotalFullBath
#asi que vamos a cambiarlo x un 0

train.loc[560, "Bedroom_TotalFullBath"] = 0


cols_to_elim = ["Unnamed: 0", "Id"]

train = train.drop(cols_to_elim, axis = 1)



X = np.array(train.drop("SalePrice", axis = 1))
Y = np.array(train["SalePrice"])

print("X_shape: " + str(X.shape))
print("Y_shape: " + str(Y.shape))

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(Y[:, np.newaxis]).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, 
                                                    test_size = 0.3, random_state = 55)



forest = RandomForestRegressor(n_estimators = 10000, 
                               criterion = "mse",
                               random_state = 55,
                               n_jobs = -1)

forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train) #esto solo lo sacamos para comprobar las diferencias 
#entre estimar en el train o el test set, es decir para ver cuanto overfitting tenemos.

y_test_pred = forest.predict(X_test)

#mse = mean_squared_error(y_test, y_pred)

#vamos a ver que ta se comporta nuestro estimador cuando los datos están escalados de esta otra forma. 

minmax_x = MinMaxScaler()
minmax_y = MinMaxScaler()

X_minmax = minmax_x.fit_transform(X)
y_minmax = minmax_y.fit_transform(Y[:, np.newaxis]).flatten()

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_minmax, y_minmax,
                                                            test_size = 0.3, random_state = 55)

forest.fit(X_train_m, y_train_m)

y_train_pred_m = forest.predict(X_train_m)

y_test_pred_m = forest.predict(X_test_m)


def nn_model(loss = "mse"):
    
    model = Sequential()
    
    model.add(
            keras.layers.Dense(
                    units = 300,
                    input_dim = X_train.shape[1],
                    kernel_initializer = 'glorot_uniform',
                    bias_initializer = 'zeros'))
    
    model.add(
            keras.layers.Dense(
                    units = 200,
                    input_dim = 300, 
                    activation = "linear"))
    
    model.add(
            keras.layers.Dense(
                    units = 1, 
                    input_dim = 200))
    
    adam = optimizers.Adam()
    
    model.compile(optimizer = adam, loss = loss)
    
    return model

model1 = nn_model()
    
model1.fit(X_train, y_train, batch_size = 200, epochs = 50, verbose = 1, validation_split = 0.1)    
    
y_pred = model1.predict(X_test)

mean_squared_error(y_test, y_pred) #parece que nuestro RandomForest funciona mejor. 





