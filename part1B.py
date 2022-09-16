import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
from f_gasbox import gasbox
from f_absolute_velocity import absolute_velocity
from f_MaxwellBoltzmann_v import MaxwellBoltzmann_v

N_H2 = 100                      # number of H_2 molecules

T = 3000                        # temperature [K]
m_H2 = const.m_H2               # mass of a H2 molecule [kg]
k = const.k_B                   # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                        # mean of our particle velocities
sigma = np.sqrt(k*T/m_H2)       # the standard deviation of our particle velocities

L = 10**(-6)                    # length of sides in box [m]
time = 10**(-9)                 # time interval for simulation [s]
steps = 1000                    # number of steps taken in simulation

analytical_mean_Ek = 1.5*k*T    # analytic solution of the mean kinetic energy of our particles [J]

r, v, count = gasbox(my, sigma, N_H2, L, time, steps)

print(f'With {N_H2:g} H2-molecules in our gasbox, {count:g} molecules hit a wall in the box during the time interval of {time:g} s')

Ek = 0
for i in range(N_H2):
    Ek += 0.5*m_H2*np.linalg.norm(v[i])**2

numerical_mean_Ek = Ek/N_H2      # numerical solution of the mean kinetic energy of our particles [J]
deviation = numerical_mean_Ek/analytical_mean_Ek*100 - 100                      
rel_err = np.abs(analytical_mean_Ek - numerical_mean_Ek)/analytical_mean_Ek

print(f'The particles have an average kinetic energy of {analytical_mean_Ek:g} when calculated analytically, and\n{numerical_mean_Ek:g} when calculated numerically') 
print(f'The numerical result deviates approximately {deviation:.2f} % from the analytical, with a relative error of {rel_err:g}')

abs_v = absolute_velocity(v, N_H2)

plt.plot(abs_v, MaxwellBoltzmann_v(m_H2, k, T, abs_v))
plt.show()

N_H2 = 10**5                     # number of H_2 molecules

r, v, count = gasbox(my, sigma, N_H2, L, time, steps)

print(f'With {N_H2:g} H2-molecules in our gasbox, {count:g} molecules hit a wall in the box during the time interval of {time:g} s')

Ek = 0
for i in range(N_H2):
    Ek += 0.5*m_H2*np.linalg.norm(v[i])**2

numerical_mean_Ek = Ek/N_H2      # numerical solution of the mean kinetic energy of our particles [J]
deviation = numerical_mean_Ek/analytical_mean_Ek*100 - 100
rel_err = np.abs(analytical_mean_Ek - numerical_mean_Ek)/analytical_mean_Ek

print(f'The particles have an average kinetic energy of {analytical_mean_Ek:g} when calculated analytically, and\n{numerical_mean_Ek:g} when calculated numerically')
print(f'The numerical result deviates approximately {deviation:.2f} % from the analytical, with a relative error of {rel_err:g}')

abs_v = absolute_velocity(v, N_H2)

plt.plot(abs_v, MaxwellBoltzmann_v(m_H2, k, T, abs_v))
plt.show()

'''
With 100 H2-molecules in our gasbox, 896 molecules hit a wall in the box during the time interval of 1e-09 s
The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
6.88348e-20 when calculated numerically
The numerical result deviates approximately 10.79 % from the analytical, with a relative error of 0.10793

With 100000 H2-molecules in our gasbox, 839009 molecules hit a wall in the box during the time interval of 1e-09 s
The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
6.22095e-20 when calculated numerically
The numerical result deviates approximately 0.13 % from the analytical, with a relative error of 0.00129253
'''
