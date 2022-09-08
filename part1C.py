import numpy as np
import ast2000tools.constants as const
from tqdm import trange
#from numba import njit

#@njit
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
                #f += m*np.linalg.norm(v[j])/dt                         
                f += m*(- v[j][2])/dt                                # updating the box's thrust force
                # spawning a new particle
                r[j] = (np.random.uniform(0, L, size = (1, 3)))      # giving it a random position within the box
                v[j] = (np.random.normal(my, sigma, size = (1, 3)))  # giving it a random velocity
            for l in range(3):
                if r[j][l] <= 0 or r[j][l] >= L:                     # checking if the particle hits one of the walls
                     v[j][l] = - v[j][l]                             # bouncing the particle back
        r += v*dt                                                    # updating the particles' velocities
    return r, v, exiting, f

N = 100                     # number of H_2 molecules

T = 3000                    # temperature [K]
m = const.m_H2              # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m)      # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10**(-9)             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

r, v, exiting, f = gasboxwnozzle(my, sigma, N, L, time, steps)

print(r)
print(v)
print(exiting)
print(f)

pp_s = exiting/time                 # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                    # the box force averaged over all time steps [N]
fl_s = pp_s*m                       # the total fuel loss per second [kg/s]

print(f'There are {pp_s:g} particles exiting the gas box per second')
print(f'The gas box exerts a thrust of {mean_f:g} N')
print(f'The box loses a mass of {fl_s:g} kg/s')

'''
SJEKK AT ALT BLIR RIKTIG / ALT ER GJORT

KJØREEKSEMPEL???

KOMMER FORDELING AV GASSPARTIKLER TIL Å FORBLI DEN SAMME OVER LENGRE TID???
'''
