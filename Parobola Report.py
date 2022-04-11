%matplotlib inline

#導入Library
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Get Data
# return Vector 分別為[初速,角度,最大垂直距離,最大水平距離]
def Parabola(V0,theta):
    g = 9.8
    Vx = V0 * math.cos( theta / 180 * math.pi ) # Vx 水平方向速度
    Vy = V0 * math.sin( theta / 180 * math.pi ) # Vy 垂直方向速度
    t = 2 * Vy / g # 時間
    H = 0.5 * g * t**2 # H 最高位置
    R = Vx * t # R 水平距離
    Vector = [V0,theta,H,R]
    return Vector

# 總資料集為Dataset，利用迴圈輸入10000組資料
# 初速為1到100隨機取數，角度為1到89隨機取數
Dataset = np.zeros((10000,4))
for i in range(0,10000,1):
    Dataset[i,:] = Parabola(random.randint(1,100),random.randint(1,89))

    
# Clean, Prepare & Manipulate Data
# 第ㄧ欄與第二欄為輸入層，第三欄與第四欄為輸出層
# 將10000筆分成8000筆訓練集與2000筆測試集
X_train = Dataset[:8000,:2]
Y_train = Dataset[:8000,2:]
X_test = Dataset[8000:,:2]
Y_test = Dataset[8000:,2:]

# Train Model
# Model共有輸入層、Hidden layer 兩層個各10個神經元，及輸出層
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 10 , activation = tf.nn.relu , input_dim = 2) ,
        tf.keras.layers.Dense(units = 10 , activation = tf.nn.relu),
        tf.keras.layers.Dense(units = 2 , activation = tf.nn.softmax)
])
model.compile(optimizer = 'adam',
              loss = 'mean_absolute_error',
              metrics = ['accuracy'])
model.summary()
model.fit(X_train , Y_train , epochs = 500 , batch_size = 1000)

score = model.evaluate(X_test , Y_test , batch_size = 1000)
print('score:',score)

Y_predict = model.predict(X_test)

plt.scatter(Y_test[:,0],Y_predict[:,0])
plt.show

plt.scatter(Y_test[:,1],Y_predict[:,1])
plt.show
