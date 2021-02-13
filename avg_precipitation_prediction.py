import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

dataset = pd.read_csv("ca2-regression.csv")


dataset = dataset.drop('record_min_temp_year', 1)
dataset = dataset.drop('record_max_temp_year', 1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('precision',3)

description = dataset.describe()
print(description)


correlations = dataset.corr() 
print(correlations)

skew_data = dataset.skew()
print(skew_data)


histogram   =  dataset.hist()

boxplot = dataset.plot(kind='box',subplots = True , layout=(3,4),sharex = False,sharey =False  )

density = dataset.plot(kind='density',subplots = True , layout=(3,4),sharex = False,sharey =False  )

correlations = dataset.corr() 
fig  = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin = -1 , vmax = 1)
fig.colorbar(cax)
plt.show()


features = dataset.iloc[:,0].values


label = dataset.iloc[:,8].values

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 0)
X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)