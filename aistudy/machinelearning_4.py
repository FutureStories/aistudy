# -*- coding: UTF-8 -*-
import h5py
import numpy as np

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
    for index in range(len(layers) - 1):
        l = index + 1
        W = np.ones((layers[index + 1], layers[index]))
        b = np.ones((layers[index + 1]))

        parameters[str(l)] = {'w':W, 'b':b}
    
    return parameters

def forwardPropagation(X, layers, parameters, Y = None, lossPrintPerNum = 100):
    '''
    前向传递缓存后向传递锁需的值
    '''
    cache = {}
    lnum = len(layers) - 1
    aPre = X
    for index in range(lnum):
        l = index + 1
        W, b = parameters[str(l)]['W'], parameters[str(l)]['b']
        z = np.dot(W, aPre) + b
        a = sigmoid(z) if lnum == l else relu(z)
        aPre = a

        cache[str(l)] = {'w': W, 'b': b, 'z':z, 'a':a}

        if(Y != None and index % lossPrintPerNum == 0 ):
            print(str(lossCrossEntropy(a, Y)))

    loss = lossCrossEntropy(aaPre, Y) if Y != None else None

    cache['0'] = {'w': None, 'a': X, }
    
    return cache,loss

def backPropagation(layers, Y, cache):
    '''
    反向梯度传递
    '''
    gradient = {}
    lnum = len(layers) - 1
    daPre = (1.0 - Y)/(1.0 - cache[str(lnum)]['a']) - Y/cache[str(lnum)]['a']
    for index in reversed(range(lnum)):
        l = index + 1
        c = cache[str(l)]
        aPre = cache[str(l-1)]['a']
        a, w, b = c['a'], c['w'], c['b']
        dz = daPre * a * (1-a)
        dw = dz * aPre
        db = dz
        daPre = dz * w

        gradient[str(l)] = {'dw':dw, 'db':db}
    
    return gradient


def lossCrossEntropy(a,y):
    '''
    交叉熵代价函数计算损失值
    '''
    return np.mean(-(y * np.log(a) + (1-y) * np.log(1-a)), axis = 1)

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
    lnum = len(layers) - 1
    parameters = initParameters(layers) #初始化参数
    cache, loss = forwardPropagation(X = X, layers = layers, parameters = parameters, Y = Y) #前向传递cache数据
     #反向梯度传递

def train():
    (trainX,trainY,testX,testY) = loadData()
    trainX,trainY,testX,testY = trainX.reshape((-1,209)),trainY.reshape((1,209)),testX.reshape((-1,50)),testY.reshape((1,50))
    parameters = multilayerBPTrain([trainX.shape[0], 6, 4, 2, trainY.shape[0]], learningRate = 0.001, trainTime = 10000)

train()