#import lib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,2].values

#train linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#polynomial Linear model (degree change for perfect fit)

from sklearn.preprocessing import PolynomialFeatures
poly_reg =PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x) 
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

# #visualising the linear regression
# plt.scatter(x, y,color='red')
# plt.plot(x,lin_reg.predict(x),color='blue') 
# plt.title('Truth or lie (Linear Regression)')
# plt.xlabel('position level')
# plt.ylabel('Salary')
# plt.show()

# # visualising the linear regression
# plt.scatter(x,y,color='red')
# plt.plot(x,lin_reg_2.predict(x_poly),color='blue') 
# plt.title('Truth or lie(poly Linear Regression)')
# plt.xlabel('position level')
# plt.ylabel('Salary')
# plt.show()

#visualising the polynomial regression results (for higher resolution and smoother)
x_grid=np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y,color='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('Truth or Bluff (polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

# Predict a new result  
lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))