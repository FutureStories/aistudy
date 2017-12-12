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

def initParameters(layers):
    parameters = {}
    for index in range(len(layers) - 1):
        l = index + 1
        parameters['W%s'%str(l)] = np.ones((layers[index + 1], layers[index]))
        parameters['b%s'%str(l)] = np.ones((layers[index + 1]))

def train():
    (trainX,trainY,testX,testY) = loadData()
    trainX,trainY,testX,testY = trainX.reshape((-1,209)),trainY.reshape((1,209)),testX.reshape((-1,50)),testY.reshape((1,50))
    parameters = initParameters([12288, 6, 4, 2, 1])

    #第一层(6个神经元)
    W1 = np.ones((6,trainX.shape[0])) #(6,12288)
    b1 = np.ones((6,1))
    l1Out = np.dot(W1, trainX) + b1 #(6,209)

    #第二层(4个神经元)
    W2 = np.ones(4,l1Out.shape[0]) #(6,4)
    b2 = np.ones((4,1))
    l2Out = np.dot(W2, l1Out) + b2 #(4,209)

    #第三层(2个神经元)
    W3 = np.ones((2,l2Out.shape[0])) #(2,4)
    b3 = np.ones((2,1))
    l3Out = np.dot(W3, l2Out) + b3 #(2,209)

    #输出层
    W4 = np.ones((1,l3Out.shape[0])) #(1,2)
    b4 = np.ones((1,1))
    z = np.dot(W4, l3Out)

    a = np.sigmoid

    print (trainX.shape)
    print (trainY.shape)
    print (testX.shape)
    print (testY.shape)

train()