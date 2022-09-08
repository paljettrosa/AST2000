import numpy as np
import ast2000tools.constants as const
from f_gasboxwnozzle import gasboxwnozzle

N_H2 = 100                  # number of H_2 molecules

T = 3000                    # temperature [K]
m_H2 = const.m_H2           # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m_H2)   # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10**(-9)             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, L, time, steps)

print(exiting)
print(f)

particles_s = exiting/time          # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                    # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2      # the total fuel loss per second [kg/s]

print(f'There are {particles_s:g} particles exiting the gas box per second')
print(f'The gas box exerts a thrust of {mean_f:g} N')
print(f'The box loses a mass of {fuel_loss_s:g} kg/s')

'''
SJEKK AT ALT BLIR RIKTIG / ALT ER GJORT

KJØREEKSEMPEL???

KOMMER FORDELING AV GASSPARTIKLER TIL Å FORBLI DEN SAMME OVER LENGRE TID???
'''
