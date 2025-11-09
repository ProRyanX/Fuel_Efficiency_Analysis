import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
fuel_analysis = pd.read_csv('fuel_efficiency_dataSet.csv')
cols = fuel_analysis.columns.tolist()
nullCols = []
for col in cols:
    if(fuel_analysis[col].isnull().sum() > 2000):
        nullCols.append(col)
print(fuel_analysis.shape)

fuel_analysis = fuel_analysis.drop(columns=nullCols, axis=1)
print(fuel_analysis.head())
print(fuel_analysis.shape)

#print(fuel_analysis.isnull().sum())