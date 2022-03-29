%matplotlib inline

#導入Library
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Get Data
def Parabola(V0,theta):
    g = 9.8
    Vx = V0 * math.cos( theta / 180 * math.pi ) # Vx 水平方向速度
    Vy = V0 * math.sin( theta / 180 * math.pi ) # Vy 垂直方向速度
    t = 2 * Vy / g # 時間
    H = 0.5 * g * t**2 # H 最高位置
    R = Vx * t # R 水平距離
    Vector = [V0,theta,H,R]
    return Vector

Dataset = np.zeros((10000,4))
for i in range(0,10000,1):
    Dataset[i,:] = Parabola(random.randint(1,100),random.randint(1,89))
print(Dataset)
# Test Data
# Improve
# 分別帶三個 Activation Function試試看 (ReLU)

# Clean, Prepare & Manipulate Data
X_train = Dataset[:8000,:2]
Y_train = Dataset[:8000,2:]
X_test = Dataset[8000:,:2]
Y_test = Dataset[8000:,2:]
print(np.shape(X_train))
print(X_train)

# Train Model
from keras.layers import Dense
model = keras.Sequential()
model.add(Dense(units = 10,
                input_shape = (2,),
                kernel_initializer = 'normal',
                activation = 'relu'))
model.add(Dense(units = 10,
                kernel_initializer = 'normal',
                activation = 'relu'))
model.add(Dense(units = 2,
                kernel_initializer = 'normal',
                activation = 'softmax'))
model.compile(optimizer = 'adam',
              loss = 'mean_absolute_error',
              metrics = ['accuracy'])
model.summary()

model.fit(X_train , Y_train , epochs = 500 , batch_size = 100)

#
score = model.evaluate(X_test , Y_test , batch_size = 100)
print('score:',score)

#
Y_predict = model.predict(X_test)
print(Y_test)
print(Y_predict)

plt.scatter(Y_test[:,0],Y_predict[:,0])
plt.show

plt.scatter(Y_test[:,1],Y_predict[:,1])
plt.show
