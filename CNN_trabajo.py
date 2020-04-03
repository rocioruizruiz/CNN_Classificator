import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import mnist
from keras.datasets import cifar10

from keras.utils.np_utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(f"{x_train.shape}")
print(f"{y_train.shape}")
print(f"{x_test.shape}")
print(f"{y_test.shape}")



x_train = x_train / 255.0
x_test = x_test / 255.0


x_test = x_test.reshape(-1,28,28,1) #canal=1 for grey scale
x_train = x_train.reshape(-1,28,28,1) #canal=1 for grey scale


y_train = to_categorical(y_train) # Encode labels to one hot vectors
y_test = to_categorical(y_test)   # Encode labels to one hot vectors


x_train.shape, x_test.shape, y_train.shape, y_test.shape


#### Data Visualitation


x_train_v = x_train.reshape(x_train.shape[0], 28, 28)

fig, axis = plt.subplots(1, 4, figsize=(20, 10))
for i, ax in enumerate(axis.flat):
    ax.imshow(x_train_v[i], cmap='binary')
    digit = y_train[i].argmax()
    ax.set(title = f"Real Number is {digit}");


# ### Normalitation

mean = np.mean(x_train)
std = np.std(x_train)

def standardize(x):
    return (x-mean)/std


epochs = 50
batch_size = 64


## CNN
#### Define model

model=Sequential()
   
model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())    
model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    
model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512,activation="relu"))
    
model.add(Dense(10,activation="softmax"))
    
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


###### Para evitar el overfitting, y aumentar la precision de los datos podriamos añadir más datos alterando los que ya tenemos con rotaciones de X grados,  zoom, desplazamiento horizontal o vertical, etc..

#### Model training

model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=10)


#### Prediction

x_test.shape

y_pred = model.predict(x_test)
x_test_v = x_test.reshape(x_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(x_test_v[i], cmap='binary')
    ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}");


pred = model.predict_classes(x_test, verbose=1)

from sklearn.metrics import accuracy_score

#vamos a convertir y_pred que esta en probabilidades a 0s y 1 para calcular el accuracy de nuestro modelo
idx = np.argmax(y_pred, axis=-1) 
idx

preds = np.zeros(y_pred.shape)

for j,i in enumerate(idx):
    preds[j,i] = 1
    
preds

accuracy_score(y_test, preds)

# El accuracy de nuestro modelo es del 99,13%
