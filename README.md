# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Import the standard libraries.

Step 2. Upload the dataset and check for any null values using .isnull() function.

Step 3. Import LabelEncoder and encode the dataset.

Step 4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

Step 5. Predict the values of arrays.

Step 6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

Step 7. Predict the values of array.

Step 8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Shashank R
RegisterNumber:  212223230205
*/
```
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"]) 
data.head()
x=data[["Position", "Level"]] 
y=data["Salary"]
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:

## MEAN SQUARED ERROR:
![image](https://github.com/user-attachments/assets/a007a7e8-1ca1-46d6-b95a-c2a3cdbb0322)

## R2 (Variance):
![image](https://github.com/user-attachments/assets/00f14cc7-844e-48bc-bc84-a85ee7cbf8cf)

## DATA PREDICTION 
![image](https://github.com/user-attachments/assets/e06402b8-1a33-466f-bba2-f18838929c7c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
