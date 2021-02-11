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