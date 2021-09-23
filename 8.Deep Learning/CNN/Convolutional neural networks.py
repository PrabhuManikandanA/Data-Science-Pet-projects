import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
#Data Preprocessing

#Preprocessing the training set (to avoid over feeding)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) 
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

#Part - 2 Building the CNN

#initialising the CNN
cnn=tf.keras.models.Sequential()

cnn.add(  tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(  tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Part -3 Flattening
cnn.add(tf.keras.layers.Flatten())

#Part- 4 full connection
cnn.add(tf.keras.layers.Dense(units=128,activation='relu',))

#part 5 OutputLayer

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#training the CNN

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

cnn.fit(x=training_set,validation_data=test_set,epochs=25)

#part - 4 
import numpy as np 
from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_7.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction ='dog'
else:
    prediction='cat'
    
print(prediction)