
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
 
#importing salary dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#splitting dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) 

#Training the simple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set result

y_pred=regressor.predict(x_test)

#Visualisation the training results
plt.scatter(x_train, y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#Visualisation the test results
plt.scatter(x_test, y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()