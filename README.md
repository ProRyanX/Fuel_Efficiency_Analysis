# 🚗 Fuel Efficiency Analysis

## 📖 Project Overview
The **Fuel Efficiency Analysis** project explores how various car features—such as engine size, horsepower, weight, cylinders, and model year—affect a vehicle’s fuel efficiency.  
This version of the project applies **Logistic Regression** to classify whether a vehicle’s **MPG (miles per gallon)** is **above or below 15**, based on its specifications. Through data cleaning, visualization, and model building, this project demonstrates how categorical prediction can be applied to automotive data.

---

## 🎯 Objectives
- Analyze and visualize the relationship between car attributes and fuel efficiency.  
- Engineer a new binary field (`efficiency_above_15`) for classification.  
- Build and train a **Logistic Regression model** to predict fuel efficiency class.  
- Evaluate the model using **accuracy, confusion matrix, and classification report**.  

---

## 🧰 Tools & Libraries
- **NumPy** – Numerical computations  
- **Pandas** – Data manipulation and preprocessing  
- **Matplotlib** & **Seaborn** – Visualization and correlation analysis  
- **Scikit-learn (sklearn)** – Label encoding, model training, and evaluation  

---

## ⚙️ Key Features
- Data cleaning with handling of missing and blank values  
- Outlier detection and removal using **IQR (Interquartile Range)**  
- Label encoding of categorical data for model compatibility  
- Creation of a derived feature `efficiency_above_15` (1 if MPG > 15, else 0)  
- Logistic Regression model trained on 80% of the data and tested on 20%  
- Visualization of the **sigmoid function** to illustrate model behavior  
- Model performance metrics including **accuracy**, **confusion matrix**, and **classification report**

---

## 📊 Results
The Logistic Regression model achieved an **accuracy of 97.9%**, correctly classifying most vehicles based on their efficiency.  
Key evaluation metrics included:  
- **Confusion Matrix:** `[[8, 2], [0, 85]]`  
- **Precision (class 1):** 0.98  
- **Recall (class 1):** 1.00  
- **F1-score (class 1):** 0.99  

These results indicate that the model effectively distinguishes between fuel-efficient and less efficient cars using the selected attributes.

---

## 🧠 Author
**Rayan Rahat**  
B.Tech in Artificial Intelligence and Data Science  
Graphic Era Deemed to be University, Dehradun  