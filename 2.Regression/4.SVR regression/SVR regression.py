#import lib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,2].values

y=y.reshape(len(y),1)




#feature scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x=sc_x.fit_transform(x)
sc_y = StandardScaler()
y=sc_y.fit_transform(y)

#import svr training the SVR model ont he whole dataset 
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

# we are using inverse since the value is reshaped
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))) 


#Visualising the SVR
x_grid=np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y),color='red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='blue')
plt.title('Truth or Bluff (Support vector regression Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show() 