import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values # features independent variables
y=dataset.iloc[:,2:].values # dependent variables

# training Decision tree 
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting a new result
regressor.predict([[6.5]])

#visualising decision tree high res
x_grid=np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth or Bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()