import numpy as np
from tqdm import trange 
#from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

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

N_H2 = 1000                 # number of H_2 molecules

T = 3000                    # temperature [K]
m = const.m_H2              # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m)      # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10**(-9)             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, L, time, steps)

pp_s = exiting/time                 # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                    # the box force averaged over all time steps [N]
fl_s = pp_s*m                       # the total fuel loss per second [kg/s])

def fuel_consumption(fuel_m, N_H2, N_b, time, sc_m, mean_f, fl_s, delta_v):
    m_H2 = const.m_H2                                    # mass of a H2 molecule [kg]
    tot_init_m = sc_m + fuel_m + m_H2*N_H2*N_b           # the rocket engine's initial mass with fuel included [kg]
    thrust_f = N_b*mean_f/time                           # the combustion chamber's total thrust force [N]
    a = thrust_f/tot_init_m                              # the rocket's acceleration [m/s**2]
    delta_t = delta_v/a                                  # time spent accelerating the rocket [s]
    tot_fl = abs(delta_t*fl_s*N_b)                       # total fuel loss [kg]
    return tot_fl

'''
let's say we want our combustion chamber (rocket engine) to be 1m x 1m x 1m.
since our boxes have a volume of 10**(-18)m**3, we need 10**18 boxes
'''

N_b = 10**18                          # number of gasboxes

print(f'There are {pp_s*N_b:g} particles exiting the combustion chamber per second')
print(f'The combustion chamber exerts a thrust of {mean_f*N_b:g} N')
print(f'The combustion chamber loses a mass of {fl_s*N_b:g} kg/s')

sc_m = mission.spacecraft_mass        # mass of rocket without fuel [kg]
fuel_m = 10**4                        # mass of feul [kg]
delta_v = 10**4                       # change in the rocket's velocity [m/s]

tot_fl = fuel_consumption(fuel_m, N_H2, N_b, time, sc_m, mean_f, fl_s, delta_v)

print(f'The rocket uses a total of {tot_fl:g} kg fuel to boost its speed {delta_v:g} m/s')

'''
FIKS FUEL_CONSUMPTION SÅNN AT MASSEN MINKER NÅR FARTEN ØKER?
'''
