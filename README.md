# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VISHAL S
RegisterNumber: 212224040364 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![422175317-1d147f40-dd3c-46c4-94d7-939a734c3609](https://github.com/user-attachments/assets/a2da0dff-c428-4f07-8dad-b929cdf23a00)
![422175509-83f96591-fda0-4889-81fe-6969515f505d](https://github.com/user-attachments/assets/4b25b31a-24e0-4614-8c40-1505dd4d2fd8)
![422175842-ba7c4be6-ed74-4d04-9d3c-3877b3dcd426](https://github.com/user-attachments/assets/b1fccec3-bf4a-4319-bca0-1b202f0d95de)
![422176043-76599986-b50b-45a5-bbb3-bd38df320c65](https://github.com/user-attachments/assets/28e07b0d-77cd-424a-b84d-1f3b88991c69)
![422176351-3f01264b-8de7-4f8a-ae11-c8f5fc67ceeb](https://github.com/user-attachments/assets/17433df2-5c3e-45a8-8fb5-a469b66c0f51)
![422176426-ae8084d1-73d2-4052-9d59-4263d00390c2](https://github.com/user-attachments/assets/af7d4ba1-7ce3-4eda-b2be-5d02a300893e)
![422176517-a8618a53-6aac-4fbe-a743-05368dc295c3](https://github.com/user-attachments/assets/fb91bfaf-81bf-40ba-952f-0ffd3458df9e)
![422176597-6ebdca72-ef31-4a92-8a28-663ffd7cb940](https://github.com/user-attachments/assets/ec80c5d9-a871-48c3-a9ff-8bddd312418c)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
