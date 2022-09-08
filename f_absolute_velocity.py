import numpy as np
from tqdm import trange

'''
made our own function to compute the absolute velocity for each particle
and sort the values from smallest to largest, because it reduced runtime
'''

def absolute_velocity(v, N_H2):
    abs_v = np.zeros(N_H2) 
    for i in trange(int(N_H2)):
        for j in range(len(v[i])):
            abs_v[i] += v[i][j]**2
        abs_v[i] = np.sqrt(abs_v[i])
    abs_v.sort()
    return abs_v
