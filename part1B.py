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

r1, v1, count1 = gasbox(my, sigma, N_H2, L, time, steps)

print(r1)
print(v1)
print(count1)

abs_v1 = absolute_velocity(v1, N_H2)

plt.plot(abs_v1, MaxwellBoltzmann_v(m_H2, k, T, abs_v1))
plt.show()

N_H2 = 10**5               # number of H_2 molecules

r2, v2, count2 = gasbox(my, sigma, N_H2, L, time, steps)

print(count2)

abs_v2 = absolute_velocity(v2, N_H2)

plt.plot(abs_v2, MaxwellBoltzmann_v(m_H2, k, T, abs_v2))
plt.show()

'''
TESTE ACCURACY?? ENDRE TIDSINTERVALL??

TA MED KJØREEKSEMPEL???
'''
