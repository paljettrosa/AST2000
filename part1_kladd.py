import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const


N = 100                 # number of H_2 molecules
T = 3000                # temperature [K]
m = const.m_H2          # mass of a H2 molecule [kg]
k = const.k_B           # Boltzmann constant [m^2*kg/s^2/K].

my = 0.0                # mean of our particle velocities
sigma = np.sqrt(k*T/m)  # the standard deviation of our particle velocities

L = 10**(-6)            # length of sides in box [m]



'''
we will now determine the coordinates where the escape hole will be positioned.
we decided for it to be located in the middle of the bottom floor of the box.
the hole is supposed to have an area of 0.25L^2 = 0.25*10**(-12). if the hole
is quadratic, each side is sqrt(0.25*10**(-12)) = 0.5*10**(-6)m
'''

'''
in the middle
'''
s = 0.5*10**(-6)        # length of each side of the escape hole
'''
corners = np.array([s/2, s/2, 0],
                   [3*s/2, s/2, 0],
                   [3*s/2, 3*s/2, 0],
                   [s/2, 3*s/2, 0])      # each corner of the escape hole's coordinates
'''

if s/2 >= r[i][0] >= 3*s/2 and s/2 >= r[i][1] >= 3*s/2 and r[i][2] == 0:
    exiting += 1
    
'''
by the corner
'''
corners = np.array([0, 0, 0],
                   [s, 0, 0],
                   [s, s, 0],
                   [0, s, 0])           # each corner of the escape hole's coordinates

for i in range(int(N)):
    for corner in corners:
        if r[i][2] == 0 and (r[i] - corner) <= corners[0]:
            exiting += 1





r =  np.random.uniform(0, L, size = (int(N), 3))         # position vector
v =  np.random.normal(my, sigma, size = (int(N), 3))     # velocity vector

'''
An array with 10 particles (such that N = 10) would look like this:

                      x =  [[x0, y0, z0],
                            [x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3],
                            [x4, y4, z4],
                            [x5, y5, z5],
                            [x6, y6, z6],
                            [x7, y7, z7],
                            [x8, y8, z8],
                            [x9, y9, z9]]
'''

#SIMULATION VARIABLES
time = 10**(-9)               # time interval for simulation [s]
steps = 1000                  # number of steps taken in simulation
dt = time/steps               # simulation step length [s]

#PREPARING THE SIMULATION
exiting = 0                   # the total number of particles that have exited the gas box
f = 0                         # used to calculate force/seconds later in the simulation

#RUNNING THE CONDITIONAL INTEGRATION LOOP
for i in range(int(steps)):
    r += v*dt

    for j in range(int(N)):
        for l in range(3):
            if r[j][l] <= 0 or r[j][l] >= L:
                v[j][l] = - v[j][l]

    '''
    To check that these conditions are fulfilled for your particles, you can use
    NumPy's logical operators, which return an array of booleans giving an
    elementwise evaluation of these conditions for an entire matrix.  You may
    need:

        (a)     np.logical_or(array1, array2)            or
        (b)     np.logical_and(array1, array2)          and
        (c)     np.less(array1, array2)                   <
        (d)     np.greater(array1, array2)                >
        (e)     np.less_equal(array1, array2)            <=
        (f)     np.greater_equal(array1, array2)         >=

    '''

    '''Now, in the same way as for the collisions, you need to count the particles
    escaping, again you can use the slow way or the fast way.
    For each particle escaping, make sure to replace it!
    Then update the total force from each of the exiting particles:
    '''
    
    #F = P*A = N*k*T*s**2
    '''
    hvis vi ser på hver partikkel
    '''
    f += m*np.linalg.norm(v[i])/dt
    f += #fill in
   


particles_per_second = exiting/time                 # the number of particles exiting per second
mean_force = f/steps                                # the box force averaged over all time steps
box_mass = particles_per_second*m                   # the total fuel loss per second

print('There are {:g} particles exiting the gas box per second.'\
.format(particles_per_second))
print('The gas box exerts a thrust of {:g} N.'.format(mean_force))
print('The box has lost a mass of {:g} kg/s.'.format(box_mass))