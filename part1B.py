import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
from f_gasbox import gasbox
from f_absolute_velocity import absolute_velocity
from f_MaxwellBoltzmann_v import MaxwellBoltzmann_v

N_H2 = 100                  # number of H_2 molecules

T = 3000                    # temperature [K]
m_H2 = const.m_H2           # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m_H2)   # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10**(-9)             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

r, v, count = gasbox(my, sigma, N_H2, L, time, steps)

print(f'With {N_H2:g} H2-molecules in our gasbox, {count:g} molecules hit a wall in the box during the time interval of {time:g} s')

abs_v = absolute_velocity(v, N_H2)

plt.plot(abs_v, MaxwellBoltzmann_v(m_H2, k, T, abs_v))
plt.show()

N_H2 = 10**5               # number of H_2 molecules

r, v, count = gasbox(my, sigma, N_H2, L, time, steps)

print(f'With {N_H2:g} H2-molecules in our gasbox, {count:g} molecules hit a wall in the box during the time interval of {time:g} s')

abs_v = absolute_velocity(v, N_H2)

plt.plot(abs_v, MaxwellBoltzmann_v(m_H2, k, T, abs_v))
plt.show()

#TESTE ACCURACY???

'''
With 100 H2-molecules in our gasbox, 883 molecules hit a wall in the box during the time interval of 1e-09 s
With 100000 H2-molecules in our gasbox, 837006 molecules hit a wall in the box during the time interval of 1e-09 s
'''
