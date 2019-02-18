import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import math
from sklearn import preprocessing


filename_rmse_matrix = str(input("Enter file name containing the rmse_matrix:\n"))
filename_length_data = str(input("Enter file name containing the length_data:\n"))

rmse_matrix = np.loadtxt(filename_rmse_matrix)
length_data = np.loadtxt(filename_length_data)

mean_column = np.mean(rmse_matrix, axis=0)	# the mean of each column
std_column = np.std(rmse_matrix, axis=0)	# standard deviation of each column

for i in range(0,len(std_column)):
	if std_column[i]>=5:
#		std_column[i] = (std_column[i]-min(std_column))/(max(std_column)-min(std_column))
		std_column[i] = 5

plt.errorbar(length_data, mean_column, yerr=std_column, color='black', ecolor='r', capsize=7, label='errors')
plt.axis(ymin=20, ymax=60, xmin=0, xmax=2500)
plt.xlabel('Data size', size=15)
plt.ylabel('Means', size=15)
# plt.title('Final Error Bar Plot')
plt.xticks(size=20)
plt.yticks(size=20)
plt.grid()
plt.legend()
plt.show()