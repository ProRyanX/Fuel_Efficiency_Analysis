import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

'''
There are somethings that have been commented out using multi-line comments. They are for examining and verification
purposes during data cleaning and analysis. Yoou can uncomment them if you want to see the intermediate outputs.
'''

# Load the dataset
fuel_analysis = pd.read_csv('fea_Dataset.csv')

# Checking for any NULL values
'''
print(fuel_analysis.isnull().sum())
'''

# Data Cleaning: Fill NULL values with the mean of the column
cols = list(fuel_analysis.columns)
for col in cols:
    if fuel_analysis[col].dtype in ['float64', 'int64']:
        fuel_analysis[col] = fuel_analysis[col].fillna(fuel_analysis[col].mean())
    else:
        fuel_analysis[col] = fuel_analysis[col].fillna(fuel_analysis[col].mode()[0])

# Verify no NULL values remain
'''
print(fuel_analysis.isnull().sum())
'''

# Outlier Detection
numeric_cols = fuel_analysis.select_dtypes(include=['float64', 'int64']).columns.tolist()
'''
for col in numeric_cols:
    plt.boxplot(fuel_analysis[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
'''

# Outlier Removal using IQR
for col in numeric_cols:
    Q1 = fuel_analysis[col].quantile(0.25)
    Q3 = fuel_analysis[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    fuel_analysis = fuel_analysis[(fuel_analysis[col] >= lower_bound) & (fuel_analysis[col] <= upper_bound)]

# Verify outliers removed
'''
for col in numeric_cols:
    plt.boxplot(fuel_analysis[col])
    plt.title(f'Boxplot of {col} after Outlier Removal')
    plt.show()
'''

# Correlation Analysis
plt.figure(figsize=(10,8))
correlation_matrix = fuel_analysis.corr(numeric_only=True)
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()

# Encoding Categorical Variables
ob_cols = fuel_analysis.select_dtypes(include=['object']).columns.tolist()
L_E = LabelEncoder()
for col in ob_cols:
    fuel_analysis[col] = L_E.fit_transform(fuel_analysis[col])
    '''
    print(f"{L_E.classes_} - {L_E.transform(L_E.classes_)}")
    '''

# Verify encoding
'''
print(fuel_analysis.head())
'''

# Checking if efficiency is greater than 15
fuel_analysis['efficiency_above_15'] = np.where(fuel_analysis['mpg'] > 15, 1, 0)

# Model Building
data_source = fuel_analysis.drop('efficiency_above_15', axis=1)
data_result = fuel_analysis['efficiency_above_15']
data_source_train, data_source_test, data_result_train, data_result_test = train_test_split(data_source,data_result,train_size=0.8,random_state=0)

# Performing Logistic Regression
Model = LogisticRegression()
Model.fit(data_source_train,data_result_train)
Predictive_result = Model.predict(data_source_test)

# Sigmoid function
def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result

# Calculate accuracy and examining model performance
accuracy = accuracy_score(data_result_test, Predictive_result) * 100
testResult_score = Model.predict_proba(data_source_test)[:,1]
'''
print(confusion_matrix(data_result_test, Predictive_result))
print(classification_report(data_result_test, Predictive_result))
'''

# Final Result Graph
x_values = np.linspace(-10,10,200)
y_values = sigmoid(x_values)
plt.figure(figsize=(8,8))
plt.plot(x_values, y_values, label='Sigmoid Function', color='red')
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Border')
plt.scatter(data_result_test, testResult_score, label='Test Data Prediction')
plt.title(f"Logistic Regression Graph -> {accuracy:.1f}% Accuracy")
plt.show()