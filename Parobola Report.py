#import Library
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Get Data
#TR_Data[V0,Vx,Vy,theta,t,hmax,R]
def Parabola(V0,theta):
    g = 9.8
    Vx = V0 * math.cos( theta / 180 * math.pi ) # Vx 水平方向速度
    Vy = V0 * math.sin( theta / 180 * math.pi ) # Vy 垂直方向速度
    t = 2 * Vy / g
    H = 0.5 * g * t**2 # H 最高位置
    R = Vx * t # R 水平距離
    Vector = [V0,Vx,Vy,theta,t,H,R]
    return Vector

TR_Data = np.zeros((101,7))
for i in range(0,101,1):
    TR_Data[i,:] = Parabola(i,random.randint(0,90))
    
print(TR_Data)
