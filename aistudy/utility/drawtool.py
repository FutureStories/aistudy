# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
def pltBeautify(xrange = None, yrange = None, xlabel = '', ylabel = ''):
    if(xrange != None):
        plt.xlim(xrange)
    if(yrange != None):
        plt.ylim(yrange)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)