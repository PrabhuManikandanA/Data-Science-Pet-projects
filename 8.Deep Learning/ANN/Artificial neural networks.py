#import datasets

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os


#Data preprocessing 

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values


#Encoding the categortical data
# Country and gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

# one hot encoding the "Geography"

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 


#Part-2 Building the ANN
#Initializing the ANN

ann = tf.keras.models.Sequential()

#Adding the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


#Adding the first hidden layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#part-3 Training the ANN 
#compliling the ANN
ann.compile(optimizer='adam', loss='binary_crossentrophy' ,metrics =['accuracy'] )

#training the ANN on the training set 
ann.fit(X_train,y_train,batch_size=32,epochs=100)

# prediction

ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))

#logical regression

y_pred=ann.predict(X_test)

y_pred=(y_pred >.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(y_test),1)),1)
