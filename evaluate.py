import numpy as np

def MSE(actual=0, forecast=0):
    return ((actual - forecast) ** 2).mean()

def MAPE(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100
