import numpy as np


def constraint(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def remap(v, x, y, clip=False):
    if x[1] == x[0]:
        return y[0]
    out = y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])
    if clip:
        out = constraint(out, y[0], y[1])
    return out
