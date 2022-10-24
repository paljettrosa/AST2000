#EGEN KODE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import jit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1A import MaxwellBoltzmann_v

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

@jit(nopython = True)
def gasbox(my, sigma, N, L, time, steps):
    
    r = np.random.uniform(0, L, size = (int(N), 3))         # position vector
    v = np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector
    dt = time/steps               # simulation step length [s]
    count = 0                     # amount of times one of the particles hit a wall
    
    for i in range(int(steps)):
        for j in range(int(N)):
            for l in range(3):
                if r[j][l] <= 0 or r[j][l] >= L:
                    count += 1
                    v[j][l] = - v[j][l]
        r += v*dt
        
    return r, v, count

'''
made our own function to compute the absolute velocity for each particle
and sort the values from smallest to largest, because it reduced runtime
'''

def absolute_velocity(v, N_H2):
    abs_v = np.zeros(N_H2) 
    for i in trange(int(N_H2)):
        for j in range(len(v[i])):
            abs_v[i] += v[i][j]**2
        abs_v[i] = np.sqrt(abs_v[i])
    abs_v.sort()
    return abs_v

@jit(nopython = True)
def gasboxwnozzle(my, sigma, N, m, L, time, steps):
    
    r = np.random.uniform(0, L, size = (int(N), 3))         # position vector
    v = np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector
    dt = time/steps                                         # simulation step length [s]
    s = np.sqrt(0.25*L**2)                                  # length of all sides of the escape hole [m]
    exiting = 0                                             # amount of particles that have exited the nozzle
    f = 0                                                   # total force from escaped particles [N] 
                 
    for i in range(int(steps)):
        for j in range(int(N)):
            if s/2 <= r[j][0] <= 3*s/2 and s/2 <= r[j][1] <= 3*s/2 and r[j][2] <= 0:
                exiting += 1                                         # counting how many particles have exited the box
                f += m*(- v[j][2])/dt                                # updating the box's thrust force
                
                '''spawning a new particle'''
                
                r[j] = (np.random.uniform(0, L, size = (1, 3)))      # giving it a random position within the box
                v[j] = (np.random.normal(my, sigma, size = (1, 3)))  # giving it a random velocity
            for l in range(3):
                if r[j][l] <= 0 or r[j][l] >= L:                     # checking if the particle hits one of the walls
                     v[j][l] = - v[j][l]                             # bouncing the particle back
        r += v*dt                                                    # updating the particles' positions
        
    return r, v, exiting, f

''' defining constants '''

T = 3000                        # temperature [K]
m_H2 = const.m_H2               # mass of a H2 molecule [kg]
k = const.k_B                   # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                        # mean of our particle velocities
sigma = np.sqrt(k*T/m_H2)       # the standard deviation of our particle velocities

L = 10**(-6)                    # length of sides in box [m]
time = 10**(-9)                 # time interval for simulation [s]
steps = 1000                    # number of steps taken in simulation

analytical_mean_Ek = 1.5*k*T    # analytic solution of the mean kinetic energy of our particles [J]

def main():
    '''
    B. Simulating Energetic Gas Particles
    '''
    N_H2 = 100                  # number of H_2 molecules
    
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
    
    N_H2 = 10**5                # number of H_2 molecules
    
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
    C. Introducinga a Nozzle
    '''
    r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)

    particles_s = exiting/time          # the number of particles exiting per second [s**(-1)]
    mean_f = f/steps                    # the box force averaged over all time steps [N]
    fuel_loss_s = particles_s*m_H2      # the total fuel loss per second [kg/s]
    
    print(f'There are {particles_s:g} particles exiting the gas box per second')
    print(f'The gas box exerts a thrust of {mean_f:g} N')
    print(f'The box loses a mass of {fuel_loss_s:g} kg/s')
    

if __name__ == '__main__':
    main()


'''
FROM B:
    
With 100 H2-molecules in our gasbox, 840 molecules hit a wall in the box during the time interval of 1e-09 s
The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
6.52807e-20 when calculated numerically
The numerical result deviates approximately 5.07 % from the analytical, with a relative error of 0.0507247

With 100000 H2-molecules in our gasbox, 838030 molecules hit a wall in the box during the time interval of 1e-09 s
The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
6.22595e-20 when calculated numerically
The numerical result deviates approximately 0.21 % from the analytical, with a relative error of 0.00209709


FROM C:

There are 3.5418e+13 particles exiting the gas box per second
The gas box exerts a thrust of 4.91288e-10 N
The box loses a mass of 1.1856e-13 kg/s
'''