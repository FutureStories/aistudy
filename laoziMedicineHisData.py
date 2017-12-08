# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import utility

X = np.array([1.0, 2.0, 3.0, 6.0, 9.0, 11.0, 12.0, 13.0, 18.0, 20.0]) #炼丹的时间(小时)
Y = np.array([1.0, 1.2, 1.7, 2.9, 4.0, 5.0,  5.0,  6.0,  7.7,  8.5]) #将颜色从红色到紫色映射到1-10内后对应的丹的颜色
plt.figure()
plt.scatter(X, Y)
utility.pltBeautify((0,25), (0,10), u'炼丹的时间(小时)', u'红色到紫色映射到1-10内后对应的丹的颜色')
plt.show()