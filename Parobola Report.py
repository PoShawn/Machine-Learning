%matplotlib inline

#導入Library
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

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

Dataset = np.zeros((100,4))
for i in range(0,100,1):
    Dataset[i,:] = Parabola(i+1,random.randint(1,89))
# Test Data
# Improve
# 分別帶三個 Activation Function試試看 (ReLU)

# Clean, Prepare & Manipulate Data
X_train = Dataset[:80,:2]
Y_train = Dataset[:80,2:]
X_test = Dataset[80:,:2]
Y_test = Dataset[80:,2:]

# Train Model
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
Dense(32, input_shape=(len(X_train),)),
Activation('relu'),
Dense(10),
Activation('softmax'),
])

model.summary()
