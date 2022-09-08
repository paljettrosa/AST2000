import numpy as np
from tqdm import trange

def absolute_velocity(v, N):
    abs_v = np.zeros(N) 
    for i in trange(int(N)):
        for j in range(len(v[i])):
            abs_v[i] += v[i][j]**2
        abs_v[i] = np.sqrt(abs_v[i])
    abs_v.sort()
    return abs_v