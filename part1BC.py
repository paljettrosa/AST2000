#EGEN KODE
#KANDIDATER 15361 & 15384
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
system = SolarSystem(83856)
mission = SpaceMission(83856)
plt.rcParams.update({'font.size': 12})

@jit(nopython = True)
def gasbox(my, sigma, N, L, time, steps):

    '''
    function for simulating the energetic gas
    in a closed gas box
    '''
    
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

    '''
    function for simulating the energetic gas
    in a gas box with nozzle
    '''

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
                
                ''' spawning a new particle '''
                
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

    counts = np.zeros(5)
    numerical_mean_Ek = np.zeros(5)
    N_H2 = 100                              # number of H_2 molecules
    for i in range(5):
        r, v, counts[i] = gasbox(my, sigma, N_H2, L, time, steps)
        
        print(f'With {N_H2:g} H2-molecules in our gasbox, {counts[i]:g} molecules hit a wall in the box during the time interval of {time:g} s')
        
        Ek = 0
        for j in range(N_H2):
            Ek += 0.5*m_H2*np.linalg.norm(v[j])**2
        
        numerical_mean_Ek[i] = Ek/N_H2      # numerical solution of the mean kinetic energy of our particles [J]                   
        rel_err = np.abs(analytical_mean_Ek - numerical_mean_Ek[i])/analytical_mean_Ek*100
        
        print(f'The particles have an average kinetic energy of {analytical_mean_Ek:g} when calculated analytically, and\n{numerical_mean_Ek[i]:g} when calculated numerically') 
        print(f'The relative error of the numerical result is approximately {rel_err:.2f}%')
    
    mean_count = np.sum(counts)/5
    deviation = (np.max(numerical_mean_Ek) - np.min(numerical_mean_Ek))/np.mean(numerical_mean_Ek)*100
    print(f"With {N_H2:g} H2-molecules in our gasbox, {mean_count:.0f} particles hit a wall during the simulation on average")
    print(f"The numerical results from 5 simulations with {N_H2:g} H2-molecules deviate approximately {deviation:.2f}%")
        
    abs_v = absolute_velocity(v, N_H2)
    
    plt.plot(abs_v, MaxwellBoltzmann_v(m_H2, k, T, abs_v), color = 'indigo')
    plt.xlabel(r'absolute velocity ($v$) [$\frac{m}{s}$]')
    plt.ylabel('probability density')
    plt.title("Maxwell-Boltzmann distribution for " + r'$v$' + f'\nwhen there are {N_H2:.0f} particles')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('100_particles.pdf')

    counts = np.zeros(5)
    numerical_mean_Ek = np.zeros(5)
    N_H2 = 10**5                            # number of H_2 molecules
    for i in range(5):
        r, v, counts[i] = gasbox(my, sigma, N_H2, L, time, steps)
    
        print(f'With {N_H2:g} H2-molecules in our gasbox, {counts[i]:g} molecules hit a wall in the box during the time interval of {time:g} s')
        
        Ek = 0
        for j in range(N_H2):
            Ek += 0.5*m_H2*np.linalg.norm(v[j])**2
        
        numerical_mean_Ek[i] = Ek/N_H2      # numerical solution of the mean kinetic energy of our particles [J]
        rel_err = np.abs(analytical_mean_Ek - numerical_mean_Ek[i])/analytical_mean_Ek*100
        
        print(f'The particles have an average kinetic energy of {analytical_mean_Ek:g} when calculated analytically, and\n{numerical_mean_Ek[i]:g} when calculated numerically')
        print(f'The relative error of the numerical result is approximately {rel_err:.2f}%')

    mean_count = np.sum(counts)/5
    deviation = (np.max(numerical_mean_Ek) - np.min(numerical_mean_Ek))/np.mean(numerical_mean_Ek)*100
    print(f"With {N_H2:g} H2-molecules in our gasbox, {mean_count:.0f} particles hit a wall during the simulation on average")
    print(f"The numerical results from 5 simulations with {N_H2:g} H2-molecules deviate approximately {deviation:.2f}%")
    
    abs_v = absolute_velocity(v, N_H2)
    
    plt.plot(abs_v, MaxwellBoltzmann_v(m_H2, k, T, abs_v), color = 'crimson')
    plt.xlabel(r'absolute velocity ($v$) [$\frac{m}{s}$]')
    plt.ylabel('probability density')
    plt.title("Maxwell-Boltzmann distribution for " + r'$v$' + f'\nwhen there are {N_H2:.0f} particles')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('100000_particles.pdf')
    
    
    
    
    '''
    C. Introducing a Nozzle
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
RESULTS:
        FROM B:    
    With 100 H2-molecules in our gasbox, 774 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    5.00472e-20 when calculated numerically
    The relative error of the numerical result is approximately 19.45%
    With 100 H2-molecules in our gasbox, 830 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    5.9331e-20 when calculated numerically
    The relative error of the numerical result is approximately 4.50%
    With 100 H2-molecules in our gasbox, 761 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    5.13783e-20 when calculated numerically
    The relative error of the numerical result is approximately 17.30%
    With 100 H2-molecules in our gasbox, 920 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    7.30773e-20 when calculated numerically
    The relative error of the numerical result is approximately 17.62%
    With 100 H2-molecules in our gasbox, 829 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    6.0329e-20 when calculated numerically
    The relative error of the numerical result is approximately 2.90%

    With 100 H2-molecules in our gasbox, 823 particles hit a wall during the simulation on average
    The numerical results from 5 simulations with 100 H2-molecules deviate approximately 39.15%

    With 100000 H2-molecules in our gasbox, 837398 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    6.21275e-20 when calculated numerically
    The relative error of the numerical result is approximately 0.00%
    With 100000 H2-molecules in our gasbox, 838382 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    6.20191e-20 when calculated numerically
    The relative error of the numerical result is approximately 0.18%
    With 100000 H2-molecules in our gasbox, 838833 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    6.23043e-20 when calculated numerically
    The relative error of the numerical result is approximately 0.28%
    With 100000 H2-molecules in our gasbox, 836705 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    6.19176e-20 when calculated numerically
    The relative error of the numerical result is approximately 0.34%
    With 100000 H2-molecules in our gasbox, 838695 molecules hit a wall in the box during the time interval of 1e-09 s
    The particles have an average kinetic energy of 6.21292e-20 when calculated analytically, and
    6.21829e-20 when calculated numerically
    The relative error of the numerical result is approximately 0.09%
    
    With 100000 H2-molecules in our gasbox, 838003 particles hit a wall during the simulation on average
    The numerical results from 5 simulations with 100000 H2-molecules deviate approximately 0.62%


        FROM C:
    There are 3.5418e+13 particles exiting the gas box per second
    The gas box exerts a thrust of 4.91288e-10 N
    The box loses a mass of 1.1856e-13 kg/s
'''