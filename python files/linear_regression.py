import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import datasets, linear_model
import random
import tkinter as tk
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import math


d = int(input("Enter number of features: "))	# dimensions
n = int(input("Enter number of data points: "))	# number of data points to be sampled
y = int(n/5)
file_name_dataset = str(input("Enter file name containing the dataset:\n"))
file_name_target = str(input("Enter file name containing the target values:\n"))

dataset = np.loadtxt(file_name_dataset)
target_value = np.loadtxt(file_name_target)

a = []
b = []
c = []
d = []
e = []

for i in range(0, y):
	a.append(i)
	b.append(i+y)
	c.append(i+(2*y))
	d.append(i+(3*y))
	e.append(i+(4*y))




set_1 = np.delete(dataset, a, 0)	# partition of dataset without the first y rows
set_2 = np.delete(dataset, b, 0)	# partition of dataset without rows y to 2*y
set_3 = np.delete(dataset, c, 0)	# partition of dataset without rows 2*y to 3*y
set_4 = np.delete(dataset, d, 0)	# partition of dataset without rows 3*y to 4*y
set_5 = np.delete(dataset, e, 0)	# partition of dataset without rows 4*y to 5*y

pred_1 = np.delete(target_value, a, 0)		# partition of target values without the first y rows
pred_2 = np.delete(target_value, b, 0)		# partition of target values without rows y to 2*y
pred_3 = np.delete(target_value, c, 0)		# partition of target values without rows 2*y to 3*y
pred_4 = np.delete(target_value, d, 0)		# partition of target values without 3*y to 4*y
pred_5 = np.delete(target_value, e, 0)		# partition of target values without rows 4*y to 5*y

mean_rmse= []

i = 0
for i in range(0, n):

	test_data = dataset[i:(i+y)]	# test dataset partion of of size y by d
	test_target = target_value[i:(i+y)]	# test target value partition of size y by d
	regr = linear_model.LinearRegression()
	j = 100
	i = i+y

	if(np.array_equal(test_data, dataset[0:y])):	# to test whether the first partition is the test dataset
		rmse_1 = []
		standard_deviation = []
		length_data = []

		while(j<=(n-y)):
			
			train_data = set_1[0:j]		# if the first partition is the test dataset, then use set_1 as training data as it does not contain the first partition (test data)
			train_target = pred_1[0:j]	# if the first partition is the test dataset, then use pred_1 as training target values as it does not contain the first partition (test target values)

			regr.fit(train_data, train_target)
			s_pred = regr.predict(test_data)	# predict target values for test data and save it into s_pred

			print('Coefficients: \n', regr.coef_)	# print coefficient matrix which contain values of parameters (theta)
			mse = mean_squared_error(test_target, s_pred)	# the mean squared error value
			rmse_1.append(math.sqrt(mse))		# the root mean squared error value

			variance = explained_variance_score(test_target, s_pred)	# explained variance score of the model
			
			print("Root Mean Square Error: \n", rmse_1)
			#print("Standard Deviation: \n", standard_deviation)
			
			length_data.append(len(train_data))		# to find the number of rows in training data after each iteration and append it to list length_data

			j = j+100


	elif(np.array_equal(test_data, dataset[y:(2*y)])):
		rmse_2 = []
		standard_deviation = []
		length_data = []

		while(j<=(n-y)):
			
			train_data = set_2[0:j]
			train_target = pred_2[0:j]

			regr.fit(train_data, train_target)
			s_pred = regr.predict(test_data)

			print('Coefficients: \n', regr.coef_)
			mse = mean_squared_error(test_target, s_pred)
			rmse_2.append(math.sqrt(mse))

			variance = explained_variance_score(test_target, s_pred)
			#standard_deviation.append(math.sqrt(variance))
			
			print("Root Mean Square Error: \n", rmse_2)
			#print("Standard Deviation: \n", standard_deviation)

			length_data.append(len(train_data))
			

			j = j+100


	elif(np.array_equal(test_data, dataset[(2*y):(3*y)])):
		rmse_3 = []
		standard_deviation = []
		length_data = []

		while(j<=(n-y)):
			
			train_data = set_3[0:j]
			train_target = pred_3[0:j]

			regr.fit(train_data, train_target)
			s_pred = regr.predict(test_data)

			print('Coefficients: \n', regr.coef_)
			mse = mean_squared_error(test_target, s_pred)
			rmse_3.append(math.sqrt(mse))

			variance = explained_variance_score(test_target, s_pred)
			#standard_deviation.append(math.sqrt(variance))

			print("Root Mean Square Error: \n", rmse_3)
			#print("Standard Deviation: \n", standard_deviation)
			
			length_data.append(len(train_data))
			

			j = j+100
		

	elif(np.array_equal(test_data, dataset[(3*y):(4*y)])):
		rmse_4 = []
		standard_deviation = []
		length_data = []

		while(j<=(n-y)):
			
			train_data = set_4[0:j]
			train_target = pred_4[0:j]

			regr.fit(train_data, train_target)
			s_pred = regr.predict(test_data)

			print('Coefficients: \n', regr.coef_)
			mse = mean_squared_error(test_target, s_pred)
			rmse_4.append(math.sqrt(mse))

			variance = explained_variance_score(test_target, s_pred)
			#standard_deviation.append(math.sqrt(variance))

			print("Root Mean Square Error: \n", rmse_4)
			#print("Standard Deviation: \n", standard_deviation)

			length_data.append(len(train_data))
			

			j = j+100
		

	elif(np.array_equal(test_data, dataset[(4*y):(5*y)])):
		rmse_5 = []
		standard_deviation = []
		length_data = []

		while(j<=(n-y)):
			
			train_data = set_5[0:j]
			train_target = pred_5[0:j]

			regr.fit(train_data, train_target)
			s_pred = regr.predict(test_data)

			print('Coefficients: \n', regr.coef_)
			mse = mean_squared_error(test_target, s_pred)
			rmse_5.append(math.sqrt(mse))

			variance = explained_variance_score(test_target, s_pred)
			#standard_deviation.append(math.sqrt(variance))
			
			print("Root Mean Square Error: \n", rmse_5)
			#print("Standard Deviation: \n", standard_deviation)

			length_data.append(len(train_data))
			

			j = j+100

rmse_1 = np.array(rmse_1)	# coverting rmse_1 from list to array 
rmse_2 = np.array(rmse_2)
rmse_3 = np.array(rmse_3)
rmse_4 = np.array(rmse_4)
rmse_5 = np.array(rmse_5)

h_1 = np.concatenate((rmse_1,rmse_2))	# concatenating rmse_1 and rmse_2
h_2 = np.concatenate((h_1,rmse_3))
h_3 = np.concatenate((h_2,rmse_4))
rmse_matrix = np.concatenate((h_3,rmse_5)) 

rmse_matrix = rmse_matrix.reshape(5,len(length_data))	# converting rmse_matrix into a 5 by ((n-y)/100) matrix

mean_column = np.mean(rmse_matrix, axis=0)	# the mean of each column
std_column = np.std(rmse_matrix, axis=0)	# standard deviation of each column

print("Column-wise mean: \n", mean_column)
print("Column-wise Standard Deviation: \n", std_column)

filename_rmse_matrix = str(input("Enter file name to save the rmse_matrix (use extension '.txt'):\n"))
filename_length_data = str(input("Enter file name to save the dataset length (use extension '.txt'):\n"))

np.savetxt(filename_length_data, length_data)
np.savetxt(filename_rmse_matrix, rmse_matrix)