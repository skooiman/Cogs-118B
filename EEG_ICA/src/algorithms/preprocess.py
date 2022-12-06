import numpy as np
import numpy.matlib

def detect_outliers(my_data):
    return my_data

def remove_drift(my_data):
    return my_data

def zero_data(my_data):
    means = np.mean(my_data, axis=1, keepdims=True)
    return my_data - np.matlib.repmat(means, 1, my_data.shape[1])  # 15)