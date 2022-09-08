import numpy as np 
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from f_gasboxwnozzle import gasboxwnozzle
from f_fuel_consumption import fuel_consumption

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

N_H2 = 1000                 # number of H_2 molecules

T = 3000                    # temperature [K]
m_H2 = const.m_H2           # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m_H2)   # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10**(-9)             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, L, time, steps)

particles_s = exiting/time          # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                    # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2      # the total fuel loss per second [kg/s]

'''
let's say we want our combustion chamber (rocket engine) to be 1m x 1m x 1m.
since our boxes have a volume of 10**(-18)m**3, we need 10**18 boxes
'''

N_box = 10**18                      # number of gasboxes

print(f'There are {particles_s*N_box:g} particles exiting the combustion chamber per second')
print(f'The combustion chamber exerts a thrust of {mean_f*N_box:g} N')
print(f'The combustion chamber loses a mass of {fuel_loss_s*N_box:g} kg/s')

spacecraft_m = mission.spacecraft_mass       # mass of rocket without fuel [kg]
fuel_m = 10**4                               # mass of feul [kg]
delta_v = 10**4                              # change in the rocket's velocity [m/s]

tot_fuel_loss = fuel_consumption(fuel_m, N_H2, N_box, time, spacecraft_m, mean_f, fuel_loss_s, delta_v)

print(f'The rocket uses a total of {tot_fuel_loss:g} kg fuel to boost its speed {delta_v:g} m/s')

'''
FIKS FUEL_CONSUMPTION SÅNN AT MASSEN MINKER NÅR FARTEN ØKER?
'''
