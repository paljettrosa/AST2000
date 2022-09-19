import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from numba import jit

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

'''
A1. Plotting the Orbits
'''

# [M] is the unit for solar mass

a = system.semi_major_axes                   # each planet's semi major axis [AU]
e = system.eccentricities                    # each planet's eccentricity
init_angles = system.initial_orbital_angles  # the angle between each planet's initial position and the x-axis
a_angles = system.aphelion_angles            # each planet's initial angle from the aphelion 

N = 1000            #amount of time steps

def plot_orbits(planets, N, a, e, init_angles, a_angles):
    x = np.zeros((len(planets), N))
    y = np.zeros((len(planets), N))
    for i in range(len(planets)):
        r = np.zeros(N)
        f = np.linspace(init_angles[i], init_angles[i] + 2*np.pi, N)
        r = a[i]*(1 - e[i]**2)/(1 + e[i]*np.cos(np.pi - a_angles[i] + f))
        x = r*np.cos(f)
        y = r*np.sin(f)
        plt.plot(x, y, color = planets[i][1], label = planets[i][0])
    plt.plot(0, 0, color = 'orange', marker = 'o', label = 'Sun')
    plt.legend()
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.title("Our planet's orbits around their sun")

#HVORDAN SKAL APHELION ANGLE IMPLEMENTERES????????????????????????????????????????????????????????????????????????????

plot_orbits(planets, N, a, e, init_angles, a_angles)

'''
A2. Simulating the Orbits
'''

M = system.star_mass                     # the sun's mass [M]

init_pos = system.initial_positions      # each planet's initial position [AU]
init_vel = system.initial_velocities     # each planet's initial velocity [AU/yr]

r0 = np.zeros((len(planets), 2))
v0 = np.zeros((len(planets), 2))
for i in range(len(planets)):
    r0[i] = np.array([init_pos[0][i], init_pos[1][i]])
    v0[i] = np.array([init_vel[0][i], init_vel[1][i]])
    
perihelion = a[0]*(1 - e[0]**2)/(1 + e[0]*np.cos(0))            # the perihelion of our home planet's orbit [AU]
aphelion = a[0]*(1 - e[0]**2)/(1 + e[0]*np.cos(a_angles[0]))    # the aphelion of our home planet's orbit [AU]                     
d = (perihelion + aphelion)/2                                   # our home planet's mean distance from the sun [AU]
P = np.sqrt(d**3/M)                                             # our home planet's revolution period [yr]

N = 30*10**4        # amount of time steps
dt = 30*P/N         # time step

@jit(nopython = True)
def simulate_orbits(planets, N, init_angles, r0, v0, M):
    G = 4*np.pi**2                          # gravitation constant [AU**3yr**(-2)M**(-1)]
    theta = np.zeros((N, len(planets)))
    r = np.zeros((N, len(planets), 2))
    v = np.zeros((N, len(planets), 2))
    theta[0] = init_angles
    r[0] = r0
    v[0] = v0
    for i in range(N - 1):
        for j in range(len(planets)):
            g = - G*M/np.linalg.norm(r[i][j])**3*r[i][j]
            v[i+1][j] = v[i][j] + g*dt/2
            r[i+1][j] = r[i][j] + v[i+1][j]*dt
            g = - G*M/np.linalg.norm(r[i+1][j])**3*r[i+1][j]
            v[i+1][j] = v[i+1][j] + g*dt/2
            theta[i+1][j] = theta[i][j] + np.linalg.norm(v[i+1][j])/np.linalg.norm(r[i+1][j])*dt
    return theta, r, v

# dropper å gange med m[i] fordi vi antar at planetene ikke trekker på sola?????????????????????????????????????????????

theta, r, v = simulate_orbits(planets, N, init_angles, r0, v0, M)

for i in range(len(planets)):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planets[i][1])
plt.show()

#FIKSE LEGEND???????????????????????????????????????????????????????????????????????????????????????????????????????????
#SKAL DE TILTE?

'''
B2: Comparing your Orbits to Kepler's Laws

Task 1
'''

N_steps = 500           # number of time steps we want to retrieve data from 

'''
we know that our planet does 20 revolutions during the simulation period, with
10000 time steps per revolution period. this means that it's about halfway done 
with the first revolution period at around the 5000th time step, and is at
this point close to the perihelion
'''

def diff_area(planets, N_steps, theta, r, v):
    a_mean_r = 0                    
    a_dtheta = 0                    
    a_mean_v = 0                    
    p_mean_r = 0                    
    p_dtheta = 0                    
    p_mean_v = 0
    for i in range(len(planets)):
        for j in range(N_steps):
            a_mean_r += np.linalg.norm(r[j][i])
            a_dtheta += theta[j][i] - theta[0][i]
            a_mean_v += np.linalg.norm(v[j][i])
            p_mean_r += np.linalg.norm(r[5000+j][i])
            p_dtheta += theta[5000+j][i] - theta[5000][i]
            p_mean_v += np.linalg.norm(v[5000+j][i])
        a_mean_r = a_mean_r/N_steps
        a_dtheta = a_dtheta/N_steps
        a_mean_v = a_mean_v/N_steps
        p_mean_r = p_mean_r/N_steps
        p_dtheta = p_dtheta/N_steps
        p_mean_v = p_mean_v/N_steps
        a_dA = 0.5*utils.AU_to_km(a_mean_r)**2*a_dtheta
        p_dA = 0.5*utils.AU_to_km(p_mean_r)**2*p_dtheta
        diff = np.abs(a_dA - p_dA)
        rel_err = diff/a_dA
        a_distance = a_mean_r*a_dtheta
        p_distance = p_mean_r*p_dtheta
        print(f'The difference between the area close to the aphelion\nand the area close to the perihelion for {planets[i][0]} is {diff:.2f} km^2\nwith a relative error of {rel_err}')
        print(f'{planets[i][0]} travelled {a_distance:.3f} AU while sweeping out the area by the aphelion\nand {p_distance:.3f} AU while sweeping out the area by the perihelion')
        print(f'{planets[i][0]} travelled with a mean velocity of {a_mean_v:.3f} AU/yr while sweeping\nout the area by the aphelion and {p_mean_v:.3f} AU/yr while sweeping out\nthe area by the perihelion\n')
        
diff_area(planets, N_steps, theta, r, v)

#HVORFOR ER DIFFERANSEN SÅ SINNSYKT STOR????????????????????????????????????????????????????????????????????????????????????????
#ER DENNE FUNKSJONEN BEDRE? HVORDAN FUNKER DEN?
'''
def area(r, P, dt):
    for i in range(system.number_of_planets):
        a1 = 1/2 * r[0:1000, i] * 2*np.pi*r[0:1000, i]/P * dt
        a2 = 1/2 * r[5000:6000, i] * 2*np.pi*r[5000:6000, i]/P * dt
        print(f'Differansen mellom de to arealene til planet {i}: {np.abs(np.sum(a1)-np.sum(a2)):16.14}   |  Den relative feilen: {np.abs(np.sum(a1)-np.sum(a2))/np.abs(np.sum(a2))}')
'''
'''
Task 2
'''

print('\n\n')

G = 4*np.pi**2                          # gravitation constant [AU**3yr**(-2)M**(-1)]
m = system.masses                       # our planet's masses [kg]

count = [[],[],[],[],[],[],[]]
Kepler_P = np.sqrt(a**3)
Newton_P = np.zeros(len(planets))
numerical_P = np.zeros(len(planets))

for i in range(len(planets)):
    for j in range(1, N):
        if np.sign(r[j][i][1]) != np.sign(r[j-1][i][1]):
            count[i].append(j)
    Newton_P[i] = np.sqrt(4*np.pi**2/(G*(M + m[i]))*a[i]**3)
    numerical_P[i] = (count[i][1] - count[i][0])*2*dt
    print(f"{planets[i][0]}: numerical approximation: {numerical_P[i]:.3f} years, Kepler's version: {Kepler_P[i]:.3f} years, Newton's version: {Newton_P[i]:.3f} years\n")

#ER IKKE KEPLERS METODE VELDIG FEIL? REGNET UT RIKTIG???????????????????????????????????????????????????????????????????????????????????????????????
#SKAL VI BRUKE SOLMASSER?
#ER NEWTON ELLER NUMERISK FASITEN?

'''
The difference between the area close to the aphelion
and the area close to the perihelion for Doofenshmirtz is 396511626568.00 km^2
with a relative error of 8.529344557359363e-06
Doofenshmirtz travelled 0.807 AU while sweeping out the area by the aphelion
and 0.807 AU while sweeping out the area by the perihelion
Doofenshmirtz travelled with a mean velocity of 4.798 AU/yr while sweeping
out the area by the aphelion and 4.801 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Blossom is 113368451379032.00 km^2
with a relative error of 0.0021162656708760097
Blossom travelled 0.697 AU while sweeping out the area by the aphelion
and 0.731 AU while sweeping out the area by the perihelion
Blossom travelled with a mean velocity of 4.143 AU/yr while sweeping
out the area by the aphelion and 4.336 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Bubbles is 252806421694032.00 km^2
with a relative error of 0.0021549760710779225
Bubbles travelled 0.336 AU while sweeping out the area by the aphelion
and 0.335 AU while sweeping out the area by the perihelion
Bubbles travelled with a mean velocity of 1.969 AU/yr while sweeping
out the area by the aphelion and 1.956 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Buttercup is 18730351296.00 km^2
with a relative error of 1.3911180733740257e-07
Buttercup travelled 0.290 AU while sweeping out the area by the aphelion
and 0.290 AU while sweeping out the area by the perihelion
Buttercup travelled with a mean velocity of 1.723 AU/yr while sweeping
out the area by the aphelion and 1.723 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Flora is 130379361793744.00 km^2
with a relative error of 0.0017035761942819421
Flora travelled 0.503 AU while sweeping out the area by the aphelion
and 0.540 AU while sweeping out the area by the perihelion
Flora travelled with a mean velocity of 2.981 AU/yr while sweeping
out the area by the aphelion and 3.199 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Stella is 35815316419792.00 km^2
with a relative error of 0.0003661256807422405
Stella travelled 0.401 AU while sweeping out the area by the aphelion
and 0.409 AU while sweeping out the area by the perihelion
Stella travelled with a mean velocity of 2.379 AU/yr while sweeping
out the area by the aphelion and 2.426 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Aisha is 73082008438720.00 km^2
with a relative error of 0.0019049158911102339
Aisha travelled 1.039 AU while sweeping out the area by the aphelion
and 1.033 AU while sweeping out the area by the perihelion
Aisha travelled with a mean velocity of 6.118 AU/yr while sweeping
out the area by the aphelion and 6.094 AU/yr while sweeping out
the area by the perihelion


Doofenshmirtz: numerical approximation: 6.739 years, Kepler's version: 11.684 years, Newton's version: 6.741 years

Blossom: numerical approximation: 9.768 years, Kepler's version: 17.675 years, Newton's version: 10.197 years

Bubbles: numerical approximation: 101.162 years, Kepler's version: 176.802 years, Newton's version: 101.994 years

Buttercup: numerical approximation: 162.160 years, Kepler's version: 279.130 years, Newton's version: 160.742 years

Flora: numerical approximation: 26.707 years, Kepler's version: 50.598 years, Newton's version: 29.192 years

Stella: numerical approximation: 59.235 years, Kepler's version: 107.177 years, Newton's version: 61.834 years

Aisha: numerical approximation: 3.651 years, Kepler's version: 6.108 years, Newton's version: 3.524 years
'''
