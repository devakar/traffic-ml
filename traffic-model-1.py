
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:18:33 2017

@author: I330519
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from numpy import log
from math import sqrt

from statsmodels.tsa.stattools import adfuller

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

import os

#data = pd.read_csv('internet-traffic-data-in-bits.csv')
#print(data.head())
#print('\n Data Types:')
#print(type(data))
#print(data['Time'].head())
#print(data['Data'].head())

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# script_dir = os.path.dirname(os.path.realpath('__file__'))
# print(script_dir) #<-- absolute dir the script is in
# rel_path = "devd/machine-learning/self/internet-traffic-data-in-bits.csv"
# abs_file_path = os.path.join(script_dir, rel_path)
# print(abs_file_path)

series = Series.from_csv('internet-traffic-data-in-bits.csv', header=0)
#series = Series.from_csv(abs_file_path, header=0)
X = series.values
#print(X)
series.plot()
pyplot.show()
series.hist()
pyplot.show()

result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value)) 
    

# X = log(X)
# pyplot.hist(X)
# pyplot.show()
# pyplot.plot(X)
# pyplot.show()

# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))

split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

#print(len(X))    
diff_values = difference(X, 1)
pyplot.plot(diff_values)
pyplot.show()
pyplot.hist(diff_values)
pyplot.show()


result = adfuller(diff_values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
   
split = int(len(diff_values) / 2)
X1, X2 = diff_values[0:split], diff_values[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
    
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
print(len(supervised_values))

# split data into train and test-sets
split = int(len(diff_values) * 0.8)
print(split)
train, test = supervised_values[0:split], supervised_values[split:]


# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
print(len(train_scaled))

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 1500, 1)

lstm_model.save('trafic_model-1.h5')

# serialize model to JSON
model_json = lstm_model.to_json()
with open("trafic_model-1json.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
lstm_model.save_weights("trafic_model-1weight.h5")
print("Saved model to disk")

del lstm_model

