import numpy as np


beta = 0.005
def Sigmoid(x):
    # 激活函数，将线性关系转化为非线性关系
    g_x = 1 / (1 + np.exp(-beta * x))
    return g_x
