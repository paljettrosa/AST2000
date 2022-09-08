import numpy as np
from tqdm import trange

def gasbox(my, sigma, N, L, time, steps):
    r = np.random.uniform(0, L, size = (int(N), 3))         # position vector
    v = np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector
    dt = time/steps               # simulation step length [s]
    count = 0                     # amount of times one of the particles hit a wall
    for i in trange(int(steps)):
        for j in range(int(N)):
            for l in range(3):
                if r[j][l] <= 0 or r[j][l] >= L:
                    count += 1
                    v[j][l] = - v[j][l]
        r += v*dt
    return r, v, count