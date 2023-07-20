#!/usr/bin/env python

# Python Linear Modeling Assignment 

# BSGP 7030

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys 

print(sys.argv)

print("Running linear modelling of data python script")
print()

if len(sys.argv) < 2:
    print("Missing filename")
    sys.exit(-1)
# Set Notebook Variables 

filename = sys.argv[1]


print("loading dataset {}".format(filename))
print()

# Read regex.csv File
dataset = pd.read_csv(filename)
print(dataset)



# Fitting Linear Regression to the Dataset 

model = LinearRegression()
model.fit(dataset[['x']], dataset [['y']])


# Visualizing the Linear Regression 

# Plot data 
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("scatterregrex.png")

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("lineregrex.png")
