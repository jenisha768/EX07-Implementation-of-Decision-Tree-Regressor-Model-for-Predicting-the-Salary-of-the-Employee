# EX 7 Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee
## DATE:
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset: Import the data (e.g., employee salary data) into a pandas DataFrame.
2. Handle Missing Values: Identify and either fill or remove missing values.
3. Encode Categorical Variables: Convert categorical columns (e.g., department, gender) into numerical form using label encoding or one-hot encoding.
4. Split the Dataset: Define your features (X) and target (y), then split the data into training and testing sets.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JENISHA TEENA ROSE F
RegisterNumber:  2305001010
*/
import pandas as pd
data=pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor
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
![image](https://github.com/user-attachments/assets/ce177f54-20a7-4b38-bb26-4aba28271786)

![image](https://github.com/user-attachments/assets/f02ecd7c-26d5-4244-928f-f5a0ab5ea726)

![image](https://github.com/user-attachments/assets/50a24b76-3a4b-4f0d-99cf-2d31860b5580)

![image](https://github.com/user-attachments/assets/fc823680-a060-4a2d-8cf3-ef5fa052f1ca)






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
