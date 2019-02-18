import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import datasets, linear_model
import random
import tkinter as tk
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import math


d = int(input("Enter number of features: "))	# dimensions
n = int(input("Enter number of data points to be sampled: "))	# number of data points to be sampled
y = int(n/5)		# size of cross validation sets


means = np.random.rand(3, d)*10	# means of the Gaussian.

covars = []
for i in range(3):
	sigma = np.random.uniform(0.1, 10, size=(d, d))
	identity_matrix = np.identity(d)
	single_covars = (sigma*sigma)*identity_matrix	# positive semi-definite covariance matrix.
	covars.append(single_covars) 	# list of 3 positive semi-definite covariance matrix, each for a single Gaussian. Here 3 since 3 Gaussians are used.

weights = np.random.dirichlet((1, 1, 1))	# sampled the weights of the Gaussian from Dirichlet distribution with alpha = [1, 1, 1]. The result is an array of dimension 1 by 3.

component1 = []
component2 = []
component3 = []

for i in range(n):
	u = np.random.uniform(0,1)
	
	if 0<=u and u<=weights[0]:
		component1.append(np.random.multivariate_normal(means[0],covars[0]))
	elif weights[0]<u and u<=weights[1]:
		component2.append(np.random.multivariate_normal(means[1],covars[1]))
	else:
		component3.append(np.random.multivariate_normal(means[2],covars[2]))

print(len(component1))
print(len(component2))
print(len(component3))


u = np.array(component1)	# for converting list of lists into array - needed for linear regression
v = np.array(component2)
w = np.array(component3)
o = np.concatenate((u,v))
dataset = np.concatenate((o,w))    # an array of 1000 by 10 (1000 samples each with 10 features)

file_name_dataset = str(input("Enter file name to save the dataset (use extension '.txt'):\n"))
np.savetxt(file_name_dataset, dataset)


# this is a random unform n by d matrix with values between 0.1 and 10. This matrix is multiplied with the dataset to obtain target value for training LR model. 
target_value = np.random.uniform(0.1, 10, size=(n,d))*dataset
file_name_target = str(input("Enter file name to save the target values (use extension '.txt'):\n"))
np.savetxt(file_name_target, target_value)