import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_startups.csv')

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

#Encoding categorical data

#Encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-1])],remainder='passthrough' )
x=np.array(ct.fit_transform(x))

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) 

#train the Multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set result

y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# # Building the optimal model using Backward Elimination
# import statsmodels.api as sm
# x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
# X_opt = x[:, [0, 1, 2, 3, 4, 5]]
# X_opt = X_opt.astype(np.float64)
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = x[:, [0, 1,2,3,4]]
# X_opt = X_opt.astype(np.float64)
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = x[:, [0, 3,2,]
# X_opt = X_opt.astype(np.float64)
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = x[:, [0, 3, 5]]
# X_opt = X_opt.astype(np.float64)
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = x[:, [0, 3]]
# X_opt = X_opt.astype(np.float64)
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()