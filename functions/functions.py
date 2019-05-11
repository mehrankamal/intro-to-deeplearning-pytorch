import numpy as np

def softmax(input):
    """
    Takes numpy array as input and returns a list of probablities of each class
    """

    return np.exp(input)/sum(np.exp(input))

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.max(0, x)
