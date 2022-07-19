# -*- coding: utf-8 -*-
"""
# DAILY BIRTH ANALYSIS
"""

#importing the necessary modules
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt

"""Reading the data from the csv file"""

#Reading the data from the csv file
dataframe=pd.read_csv("/daily-total-female-births.csv")
#printing the first 20 datas
dataframe.head(20)

"""Calulating the diffrence between the i th data and i-1 th data  """

#Calculating the difference between i th data and i-1 th data
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)

X = difference(dataframe['Births'])
size = int(len(X) * 0.66)
print("Size=",size)

"""AUTOCORRELATION:
     Autocorrelation represents the degree of similarity between a given time series and a lagged version of itself over successive time intervals. Autocorrelation measures the relationship between a variable's current value and its past values.
     
     POSITIVE CORRELATION :0  < Correlation value < 1
     NEGATIVE CORRELATION :-1 < Correlation value < 0
     
LEVELS OF CORRELATION:

     STRONG POSITIVE CORRELATION: Correlation Value should be greater than 0.5
     STRONG NEGATIVE CORRELATION: Correlation Value should be less than -0.5
     
     MODERATE POSITIVE CORRELATION: Correlation value is in the range of 0.3 to 0.49
     MODERATE NEGATIVE CORRELATION: Correlation value is in the range of -0.3 to -0.49
     
     WEAK POSITIVE CORRELATION: Correlation value is in the range of 0.2 to 0.39
     WEAK NEGATIVE CORRELATION: Correlation value is in the range of -0.2 to -0.39
     
"""

#autocorrelation plot
fig, ax = plt.subplots(figsize=(16,8))
plot_acf(X, lags=50, ax=ax)
plt.ylim([0,1])
plt.yticks(np.arange(-1.1, 1.1, 0.1))
plt.xticks(np.arange(1, 51, 1))
plt.axhline(y=0.5, color="green")
plt.axhline(y=0.3, color="red")
plt.axhline(y=-0.5, color="green")
plt.axhline(y=-0.3, color="red")
plt.show()

#spliting the data into train and test sets
train, test = X[0:size], X[size:]

"""AUTOREGRESSION:
      Autoregression is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step.

Building the AutoRegressive Model
"""

#Building the model
window = 6
model = AutoReg(train, lags=21)
model_fit = model.fit()
coef = model_fit.params
print(coef)

#prediction 
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
        #print(i," i")
        #print(coef[i],"*",history[-i],"=",yhat)
    return yhat

"""Predicting the values for the test set"""

history = [train[i] for i in range(len(train))]
#print(history)
predictions = list()
for t in range(len(test)):
    yhat = predict(coef,history)
    #print(yhat)
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    #print(history)

"""Root Mean Square Error Calculation"""

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

"""Plotting the original values"""

pyplot.plot(test)

"""Plotting the predicted values"""

pyplot.plot(predictions)

pyplot.plot(test)
pyplot.plot(predictions,color='red')
pyplot.show()