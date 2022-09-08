import numpy as np
from tqdm import trange
import ast2000tools.constants as const

def gasboxwnozzle(my, sigma, N, L, time, steps):
    r = np.random.uniform(0, L, size = (int(N), 3))         # position vector
    v = np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector
    m = const.m_H2                # mass of a H2 molecule [kg]
    dt = time/steps               # simulation step length [s]
    s = np.sqrt(0.25*L**2)        # length of all sides of the escape hole [m]
    exiting = 0                   # amount of particles that have exited the nozzle
    f = 0                         # total force from escaped particles [N]                  
    for i in trange(int(steps)):
        for j in range(int(N)):
            if s/2 <= r[j][0] <= 3*s/2 and s/2 <= r[j][1] <= 3*s/2 and r[j][2] <= 0:
                exiting += 1                                         # counting how many particles have exited the box
                f += m*(- v[j][2])/dt                                # updating the box's thrust force
                # spawning a new particle
                r[j] = (np.random.uniform(0, L, size = (1, 3)))      # giving it a random position within the box
                v[j] = (np.random.normal(my, sigma, size = (1, 3)))  # giving it a random velocity
            for l in range(3):
                if r[j][l] <= 0 or r[j][l] >= L:                     # checking if the particle hits one of the walls
                     v[j][l] = - v[j][l]                             # bouncing the particle back
        r += v*dt                                                    # updating the particles' velocities
    return r, v, exiting, f