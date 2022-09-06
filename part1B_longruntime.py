import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
from tqdm import trange

def gasbox(my, sigma, N, L, time, steps):
    r =  np.random.uniform(0, L, size = (int(N), 3))         # position vector
    v =  np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector
    abs_v = np.zeros(int(N))
    count = 0                     # amount of times one of the particles hit a wall
    dt = time/steps               # simulation step length [s]
    for i in trange(int(steps)):
        r += v*dt
        for j in range(int(N)):
            for l in range(3):
                if r[j][l] <= 0 or r[j][l] >= L:
                    count += 1
                    v[j][l] = - v[j][l]
            abs_v[j] = np.linalg.norm(v[j])
    abs_v.sort()
    return r, v, abs_v, count

N = 100                     # number of H_2 molecules
T = 3000                    # temperature [K]
m = const.m_H2              # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m)      # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10**(-9)             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

r1, v1, abs_v1, count1 = gasbox(my, sigma, N, L, time, steps)

print(r1)
print(v1)
print(abs_v1)
print(count1)

def MaxwellBoltzmann_v(m, k, T, v):
    return (m/(2*np.pi*k*T))**(3/2)*np.exp(-(1/2)*(m*v**2/(k*T)))*4*np.pi*v**2

plt.plot(abs_v1, MaxwellBoltzmann_v(m, k, T, abs_v1))
plt.show()

N = 10**5               # number of H_2 molecules

r2, v2, abs_v2, count2 = gasbox(my, sigma, N, L, time, steps)

print(count2)

plt.plot(abs_v2, MaxwellBoltzmann_v(m, k, T, abs_v2))
plt.show()