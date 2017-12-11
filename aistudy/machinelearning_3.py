# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import utility.drawtool as drawt

X = np.array([1.0, 2.0, 3.0, 6.0, 9.0, 11.0, 12.0, 13.0, 18.0, 20.0]) #炼丹的时间(小时)
Y = np.array([1.0, 1.2, 1.7, 2.9, 4.0, 5.0,  5.0,  6.0,  7.7,  8.5]) #将颜色从红色到紫色映射到1-10内后对应的丹的颜色

def hisDataShow():
    '''
    将历史数据在直角坐标轴上显示
    '''
    plt.figure()
    plt.scatter(X, Y)
    drawt.pltBeautify((0,25), (0,10), u'炼丹的时间(小时)', u'红色到紫色映射到1-10内后对应的丹的颜色')
    plt.show()

def calLossWithABSError():
    '''
    当w=1, b=1时的绝对值误差
    '''
    w = 1
    b = 1
    a = w * X + b
    result = np.sum(np.abs(Y-a))
    print (str(Y))
    print (str(a))
    print ('绝对值误差为%s'%result)

def calLossWithSquareError():
    '''
    当w=1, b=1时的平方误差
    '''
    w = 1
    b = 1
    a = w * X + b
    result = np.sum(np.square(Y-a))
    print (str(Y))
    print (str(a))
    print ('平方误差为%s'%result)

def oneLayerBPNet():
    X_in, Y_out = X.reshape((1,-1)), Y.reshape((1,-1))
    W,b,learnRate = np.array([[1]]), 1, 0.001
    for _ in range(0,10000):
        Z = np.dot(W, X_in) + b

        loss_1 = Z - Y_out
        loss,dz = np.sum(np.square(loss_1)),2 * loss_1

        b = b - learnRate * np.mean(dz)
        W = W - learnRate * np.mean(dz * X_in, axis = 1).T

        if(_ % 100 == 0):
            print ('loss = %s W = %s b = %s'%(loss, str(W), b))
    
    xDraw = np.arange(X_in[0][0] ,  X_in[0][-1], 0.1).reshape((1,-1))
    yDraw = np.dot(W, xDraw) + b
    plt.figure()
    plt.scatter(X_in, Y_out, c = 'b')
    plt.scatter(xDraw, yDraw, c = 'r')
    plt.show()

#hisDataShow()
#calLossWithABSError()
#calLossWithSquareError()
oneLayerBPNet()