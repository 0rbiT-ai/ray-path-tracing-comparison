import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def dot(a, b):
    return np.dot(a, b)