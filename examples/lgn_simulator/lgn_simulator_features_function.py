import numpy as np

def irfmax(t, U):
    return None, U[0]

def irfmin(t, U):
    return None, U[1:].min()

def irf_size(t, U):
    return None, t[np.argmin(U[1:])]
