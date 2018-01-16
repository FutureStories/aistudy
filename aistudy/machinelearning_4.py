# -*- coding: UTF-8 -*-
import h5py
import numpy as np
import utility.drawtool as drawt
import matplotlib.pyplot as plt

def loadData():
    '''
    加载猫咪图片
    '''
    fTrainData = h5py.File('./catedatasets/train_catvnoncat.h5', 'r')
    fTestData = h5py.File('./catedatasets/test_catvnoncat.h5', 'r')
    
    trainX = np.array(fTrainData['train_set_x'][:])
    trainY = np.array(fTrainData['train_set_y'][:])

    testX = np.array(fTestData['test_set_x'][:])
    testY = np.array(fTestData['test_set_y'][:])
    
    return trainX,trainY,testX,testY

def sigmoid(input):
    return 1.0/(1.0 + np.exp(-input))

def relu(input):
    return np.maximum(0,input)

def initParameters(layers):
    '''
    初始化parameters
    layers: 网络结构
    
    return: parameters保存了网络每一层的w、b值。
            数据结构:{'1':{'w':[[Array]], 'b':[[Array]]}, '2':{'w':[[Array]], 'b':[[Array]]}}
    '''
    parameters = {}
    for l in range(1, len(layers)):
        w = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
        b = np.zeros((layers[l], 1))

        parameters[str(l)] = {'w':w, 'b':b}
    
    return parameters

def forwardPropagation(X, layers, parameters, Y = None):
    '''
    前向传递
    
    X: 输入层
    
    layers: 网络结构
            [100,8,4,1]代表两个隐藏层的神经网络
            输入层有10个神经元，第一和第二隐藏层分别有8、4个神经元，输出层有1个神经元
    
    parameters: 保存了网络每一层的w、b值。
                数据结构:{'1':{'w':[[Array]], 'b':[[Array]]}, '2':{'w':[[Array]], 'b':[[Array]]}}
    
    Y: 真实值
    '''
    cache = {}
    lnum = len(layers) - 1
    aPre = X
    for l in range(1, len(layers)):
        w, b = parameters[str(l)]['w'], parameters[str(l)]['b']
        z = np.dot(w, aPre) + b
        a = sigmoid(z) if lnum == l else relu(z)
        aPre = a

        cache[str(l)] = {'w': w, 'b': b, 'z':z, 'a':a}

    lose = lossCrossEntropy(aPre, Y) if Y != None else None

    cache['0'] = {'w': None, 'a': X, }
    
    return cache,lose

def lossCrossEntropy(a, y):
    '''
    交叉熵代价函数计算损失值
    '''
    return np.squeeze(np.mean(-(y * np.log(a) + (1.0 - y) * np.log(1.0 - a))))

def backPropagation(layers, Y, cache):
    '''
    反向梯度传递

    layers: 网络结构
        [100,8,4,1]代表两个隐藏层的神经网络
        输入层有10个神经元，第一和第二隐藏层分别有8、4个神经元，输出层有1个神经元
    
    Y: 真实值

    cache: 前向传递的输出值

    '''
    gradient = {}
    lnum = len(layers) - 1
    da = (1.0 - Y)/(1.0 - cache[str(lnum)]['a']) - Y/cache[str(lnum)]['a']
    for l in reversed(range(1, len(layers))):
        c = cache[str(l)]
        aPre = cache[str(l-1)]['a']
        m = aPre.shape[1] * 1.0
        a, w, b, z = c['a'], c['w'], c['b'], c['z']
        dz = da * a * (1-a) if l == lnum else da
        if(l != lnum):
            dz[z<= 0] = 0
        dw = np.dot(dz, aPre.T) / m 
        db = np.mean(dz, axis = 1, keepdims = True)
        da = np.dot(w.T, dz)

        gradient[str(l)] = {'dw':dw, 'db':db}
    
    return gradient

def updateParameter(layers, parameters, gradient, learningRate):
    for l in range(1, len(layers)):
        grad = gradient[str(l)]
        dw, db = grad['dw'], grad['db']
        parameters[str(l)]['w'] = parameters[str(l)]['w'] - learningRate * dw
        parameters[str(l)]['b'] = parameters[str(l)]['b'] - learningRate * db
    return parameters

def multilayerBPTrain(X, Y, layers, learningRate, trainTime):
    '''
    多层神经网络训练入口
    layers: 网络结构
            [100,8,4,1]代表两个隐藏层的神经网络
            输入层有10个神经元，第一和第二隐藏层分别有8、4个神经元，输出层有1个神经元
    learningRate: 学习率
    trainTime: 训练次数

    return: 每层神经网络的w和b
    '''
    parameters = initParameters(layers) #初始化参数
    for _ in range(trainTime):
        cache, lose = forwardPropagation(X = X, layers = layers, parameters = parameters, Y = Y) #前向传递cache数据
        gradient = backPropagation(layers, Y, cache) #反向梯度传递
        parameters = updateParameter(layers, parameters, gradient, learningRate)
        if(_%100 == 0):
            print('lose = %s'%str(lose))
    
    return parameters

def verify(layers, paramaters, X, Y):
    cache, lose = forwardPropagation(X, layers, paramaters, Y)
    a = cache[str(len(layers) - 1)]['a']
    a[a >= 0.5] = 1
    a[a < 0.5] = 0
    return 1.0 - np.mean(np.abs(a - Y))

def train():
    (trainX,trainY,testX,testY) = loadData()
    trainX,trainY,testX,testY = trainX.reshape(209, -1).T,trainY.reshape(1, 209),testX.reshape(50, -1).T,testY.reshape(1, 50)
    trainX, testX = trainX/255.0, testX/255.0
    layers = [trainX.shape[0], 6, 4, 2, trainY.shape[0]]
    parameters = multilayerBPTrain(trainX, trainY, layers, learningRate = 0.01, trainTime = 4000)
    accuracy = verify(layers, parameters, trainX, trainY)
    print('accuracy = %s'%accuracy)

def sigmoidShow():
    x = np.arange(-10.0, 10.0, 0.1)
    y = sigmoid(x)
    plt.figure()
    plt.scatter(x, y)
    drawt.pltBeautify((-10, 10), (0, 1), 'X', 'sigmoid(X)')
    plt.show()

def reluShow():
    x = np.arange(-10.0, 10.0, 0.1)
    y = relu(x)
    plt.figure()
    plt.scatter(x, y)
    drawt.pltBeautify((-10, 10), (0, 10), 'X', 'relu(X)')
    plt.show()

#train()
reluShow()