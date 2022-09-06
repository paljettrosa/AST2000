import numpy as np
import ast2000tools.constants as const
#from ast2000tools.space_mission import spacecraft_mass
from tqdm import trange

def gasboxwnozzle(my, sigma, N, L, time, steps):
    r = np.random.uniform(0, L, size = (int(N), 3))         # position vector
    v = np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector
    dt = time/steps               # simulation step length [s]
    s = np.sqrt(0.25*L**2)        # length of all sides of the escape hole [m]
    count = 0                     # amount of times one of the particles hit a wall
    exiting = 0                   # amount of particles that have exited the nozzle
    f = 0                         # total force from escaped particles [N]
    for i in trange(int(steps)):
        for j in range(int(N)):
            if s/2 <= r[j][0] <= 3*s/2 and s/2 <= r[j][1] <= 3*s/2 and r[j][2] <= 0:
                exiting += 1
                f += m*np.linalg.norm(v[j])/dt                          #riktig for kraften?
                r[j] = (np.random.uniform(0, L, size = (1, 3)))         #riktig for å erstatte? tilfeldig pos?
                v[j] = (np.random.normal(my, sigma, size = (1, 3)))
            else:
                for l in range(3):
                    if r[j][l] <= 0 or r[j][l] >= L:
                        count += 1
                        v[j][l] = - v[j][l]
        r += v*dt
    cps = exiting*m/time          # fuel consumption per second [kg/s]
    return r, v, count, exiting, f, cps

N = 100                     # number of H_2 molecules

T = 3000                    # temperature [K]
m = const.m_H2              # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

my = 0.0                    # mean of our particle velocities
sigma = np.sqrt(k*T/m)      # the standard deviation of our particle velocities

L = 10**(-6)                # length of sides in box [m]
time = 10             # time interval for simulation [s]
steps = 1000                # number of steps taken in simulation

'''
let's say we want our combustion chamber (rocket engine) to be 1m x 1m x 1m.
since our boxes have a volume of 10**(-18)m**3, we need 10**18 boxes
'''

def fuel_consumption(N_H2, N_b, f, cps, init_m, delta_v):
    m_H2 = const.m_H2                       # mass of a H2 molecule [kg]
    tot_init_m = init_m + N_b*N_H2*m_H2     # the rocket engine's initial mass with fuel included [kg]
    thrust_f = N_b*f                        # the combustion chamber's total thrust force [N]
    a = thrust_f/tot_init_m                 # the rocket's acceleration [m/s**2]
    time = delta_v/a                        # time spent accelerating the rocket [s]
    tot_fc = time*cps                       # total fuel consumption [kg]
    return tot_fc

r, v, count, exiting, f, cps = gasboxwnozzle(my, sigma, N, L, time, steps)

N_b = 10**18                    # number of gasboxes
#init_m = spacecraft_mass       # mass of rocket without fuel [kg]
init_m = 1000                   # mass of rocket without fuel [kg]
delta_v = 10**3                 # change in the rocket's velocity [m/s]

tot_fc = fuel_consumption(N, N_b, f, cps, init_m, delta_v)

print(tot_fc)

'''
HVORDAN BRUKE SPACECRAFT_MASS???

HVORFOR BLIR TOT_FC SÅ LAVT TALL???
'''
#
'''
The next step is to determine our rocket engine’s performance. As explained in the lecture notes, the actual
rocket engine is a superposition of many small rocket
engines. The question now is how many small rocket engines will you need? Well, it depends... For now, make
an educated guess and be prepared to change this later.
Write your code with this in mind!
The goal of this challenge is to write a program that
determines the amount of fuel your rocket will burn for an
arbitrary boost ∆v. The program should have a structure
similar to this:
input
rocket thrust force (N)
 fuel consumption (kg/s)
initial rocket mass (kg)
speed boost (m/s)
output
fuel consumed (kg)
Remember to consider that the rocket’s total mass is split
between its fuel and the mass of the spacecraft itself.
The latter can be obtained from your instance of the
SpaceMission class (you should have looked at the documentation for the ast2000tools package by now). You
do not need to take gravity into account in this challenge,
you are just testing your rocket’s potential.
Later in the project when we are going to use this
program to calculate our fuel consumption during the
journey. We are also going to ignore the time we spend
accelerating (in this program) as it is miniscule compared
to the time it takes to travel between planets.
'''