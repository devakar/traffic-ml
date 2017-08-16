from keras.models import load_model
import os
import numpy as np
from pandas import Series
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from numpy import log
from math import sqrt

# scale train and test data to [-1, 1]
def scale(train, test):
    	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(-1, 1)
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(-1, 1)
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

def inverse_difference(history, yhat, interval=1):
    	return yhat + history[-interval]


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]



series = Series.from_csv('internet-traffic-data-in-bits.csv', header=0)

X = series.values

x_diff = difference(X,1)

x_diff = x_diff.values

l = len(x_diff)

split = int(l*0.8)
#split = 1

#train = x_diff[:split]
#test = x_diff[split:]

#train = x_diff[:-100]
#test = x_diff[-100:]

train = x_diff[:split]
test = x_diff[split:]

scaler, train_scaled, test_scaled = scale(train, test)

model = load_model('trafic_model-1.h5')

predictions = list()
expectation = list()
for i in range(len(test_scaled)):
    x = test_scaled[i]
    yhat = forecast_lstm(model, 1, x)
    yhat = invert_scale(scaler, x, yhat)
    #yhatarr = np.array(yhat)
    #yhat = scaler.inverse_transform(yhatarr)
    yhat = inverse_difference(X, yhat, len(test_scaled)+1-i)
    #yhat = inverse_difference(X, yhat, len(test_scaled)+1-i)
    # if yhat < 0:
    #     yhat = X[len(train)+1+i]
    predictions.append(yhat)
    #expected = test_scaled[i]
    expected = X[len(train)+1+i]
    expectation.append(expected)
    print('time=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

rmse = sqrt(mean_squared_error(expectation, predictions))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(expectation)
pyplot.plot(predictions)
pyplot.savefig('pred1-model-1.png')
pyplot.show()

mae = mean_absolute_error(expectation, predictions)
print("TEST MAE: %.3f" %mae)

mape = mae *100
print("Test MAPE: %.3f" %mape)

# mape2 = mean_absolute_percentage_error(y_regression_test, preds)
# print("Test MAPE: %.3f" %mape2)

#print("Test mean_absolute_percentage_error", np.mean(np.abs((expectation - predictions) / expectation)) * 100)

r_square = r2_score(expectation, predictions)
print("Test R_square %.3f" %r_square)

expectation = np.array(expectation)
predictions = np.array(predictions)
mse = np.sum(abs(expectation - predictions) / expectation) / len(expectation)
print("Test MSE: %.3f" % mse)
pmse = mse *100
print("Test PSME: %.3f" %pmse) 


print("Test mean_absolute_percentage_error", np.mean(np.abs((expectation - predictions) / expectation)) * 100)


# count_5 = 0
# count_10=0
# count_15=0
# count_20=0
# count_25=0
# count_30=0
# count_35=0
# count_40=0
# count_1 = 0
# count_90 = 0
# count_5_90 =0
# count_10_90 =0
# count_15_90 =0
# count_20_90 =0
# count_25_90 =0
# count_30_90 =0
# for i in range(len(expectation)):
# 	if expectation[i]>1:
# 		count_1 = count_1 +1
# 	if expectation[i]>1 and  abs(expectation[i]-predictions[i]) > (0.05*expectation[i]):
# 		count_5 = count_5 + 1
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.1*expectation[i]):
# 		count_10 = count_10 + 1
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.15*expectation[i]):
# 		count_15 = count_15 + 1
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.2*expectation[i]):
# 		count_20 = count_20 + 1 
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.25*expectation[i]):
# 		count_25 = count_25 + 1 
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.3*expectation[i]):
# 		count_30 = count_30 + 1 
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.35*expectation[i]):
# 		count_35 = count_35 + 1 
# 	if expectation[i]>1 and abs(expectation[i]-predictions[i]) > (0.4*expectation[i]):
# 		count_40 = count_40 + 1
# 	if expectation[i] > 90:
# 		count_90 =count_90 + 1
# 		if abs(expectation[i]-predictions[i]) > (0.05*expectation[i]):
# 			count_5_90 = count_5_90  + 1
# 		if abs(expectation[i]-predictions[i]) > (0.10*expectation[i]):
# 			count_10_90 = count_10_90  + 1
# 		if abs(expectation[i]-predictions[i]) > (0.15*expectation[i]):
# 			count_15_90 = count_15_90  + 1
# 		if abs(expectation[i]-predictions[i]) > (0.20*expectation[i]):
# 			count_20_90 = count_20_90  + 1
# 		if abs(expectation[i]-predictions[i]) > (0.25*expectation[i]):
# 			count_25_90 = count_25_90  + 1
# 		if abs(expectation[i]-predictions[i]) > (0.3*expectation[i]):
# 			count_30_90 = count_30_90  + 1


# print("length of expectation array: ", len(expectation))
# print("data with less than 1 percent", count_1)
# print("Error with more than 5 percent", count_5)
# print("Error with more than 10 percent", count_10)
# print("Error with more than 15 percent", count_15)
# print("Error with more than 20 percent", count_20)
# print("Error with more than 25 percent", count_25)
# print("Error with more than 30 percent", count_30)
# print("Error with more than 35 percent", count_35)
# print("Error with more than 40 percent", count_40)

# print("data with more than 90 percent", count_90)
# print("Error with more than 5 percent when data value is more than 90", count_5_90)
# print("Error with more than 5 percent when data value is more than 90", count_10_90)
# print("Error with more than 5 percent when data value is more than 90", count_15_90)
# print("Error with more than 5 percent when data value is more than 90", count_20_90)
# print("Error with more than 5 percent when data value is more than 90", count_25_90)
# print("Error with more than 5 percent when data value is more than 90", count_30_90)

# print("percentage of wrong answer within 5 percent error", (count_5 *100)/len(expectation))
# print("percentage of wrong answer within 10 percent error", (count_10 *100)/len(expectation))
# print("percentage of wrong answer within 15 percent error", (count_15 *100)/len(expectation))
# print("percentage of wrong answer within 20 percent error", (count_20 *100)/len(expectation))
# print("percentage of wrong answer within 25 percent error", (count_25 *100)/len(expectation))
# print("percentage of wrong answer within 30 percent error", (count_30 *100)/len(expectation))
# print("percentage of wrong answer within 35 percent error", (count_35 *100)/len(expectation))
# print("percentage of wrong answer within 40 percent error", (count_40 *100)/len(expectation))

# print("percentage of wrong spike prediction when data is 90, within 5 percent error", (count_5_90*100)/count_90 )
# print("percentage of wrong spike prediction when data is 90, within 10 percent error", (count_10_90*100)/count_90 )
# print("percentage of wrong spike prediction when data is 90, within 15 percent error", (count_15_90*100)/count_90 )
# print("percentage of wrong spike prediction when data is 90, within 20 percent error", (count_20_90*100)/count_90 )
# print("percentage of wrong spike prediction when data is 90, within 25 percent error", (count_25_90*100)/count_90 )
# print("percentage of wrong spike prediction when data is 90, within 30 percent error", (count_30_90*100)/count_90 )