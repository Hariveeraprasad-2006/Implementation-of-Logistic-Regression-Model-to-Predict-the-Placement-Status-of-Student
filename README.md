# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the python library pandas
2.Read the dataset of Placement_Data
3.Copy the dataset in data1
4.Remove the columns which have null values using drop()
5.Import the LabelEncoder for preprocessing of the dataset
6.Assign x and y as status column values
7.From sklearn library select the model to perform Logistic Regression
8.Print the accuracy, confusion matrix and classification report of the dataset

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Arikatla Hari Veera Prasad
RegisterNumber:  212223240014
*/
```
```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
```
```
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1) # Removes the specified row or column
data1.head()
```
```
data1.isnull().sum()
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```
```
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
```
```
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver ='liblinear') # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) # Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) ,True +ve/
#accuracy_score (y_true,y_pred, normalize = false)
# Normalize : It contains the boolean value (True/False). If False, return the number of correct
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
`````

## Output:
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/d56abad8-a015-4f96-b41a-1004fb622d2b)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/a95b0bbc-51f7-4572-8be6-4318bf45d106)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/000b1d68-fe35-4c58-9a19-d2fec2ae90b4)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/47de1c51-1df9-46e6-9edb-00631089ba72)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/77555fdb-d602-4e09-bf6b-03e5bd9c2e2f)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/22b3c9f4-3279-4449-a31c-c2e7bd4b192f)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/99849322-f423-495a-9018-66665f637ce9)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145049988/7d606de6-6434-4cf3-a448-433e4b588ebb)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
