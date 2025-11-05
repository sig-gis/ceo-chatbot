import numpy as np

def a_b():
    a = np.arange(100).reshape(10,10)
    b = np.arange(100,200).reshape(10,10)
    return a+b