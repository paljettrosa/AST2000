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

@jit(nopython = True)
def plot_orbits(planets, N, a, e, init_angles, a_angles):
    x = np.zeros((len(planets), N))
    y = np.zeros((len(planets), N))
    for i in range(len(planets)):
        r = np.zeros(N)
        f = np.linspace(init_angles[i], init_angles[i] + 2*np.pi, N)
        r = a[i]*(1 - e[i]**2)/(1 + e[i]*np.cos(np.pi - a_angles[i] + f))
        x[i] = r*np.cos(f)
        y[i] = r*np.sin(f)
    return x, y

x, y = plot_orbits(planets, N, a, e, init_angles, a_angles)

for i in range(len(planets)):
    plt.plot(x[i], y[i], color = planets[i][1], label = planets[i][0])

'''
plt.plot(0, 0, color = 'orange', marker = 'o', label = 'Sun')
plt.legend()
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our planet's orbits around their sun")
plt.show()
'''

'''
A2. Simulating the Orbits
'''

M = system.star_mass                     # the sun's mass [M]

init_pos = system.initial_positions      # each planet's initial position [AU/yr]
init_vel = system.initial_velocities     # each planet's initial velocity [AU/yr]

r0 = np.zeros((len(planets), 2))
v0 = np.zeros((len(planets), 2))
for i in range(len(planets)):
    r0[i] = np.array([init_pos[0][i], init_pos[1][i]])
    v0[i] = np.array([init_vel[0][i], init_vel[1][i]])
    
perihelion = a[0]*(1 - e[0]**2)/(1 + e[0]*np.cos(0))            # the perihelion of our home planet's orbit [AU]
aphelion = a[0]*(1 - e[0]**2)/(1 + e[0]*np.cos(a_angles[0]))    # the aphelion of our home planet's orbit [AU]
d = np.linalg.norm([perihelion, aphelion])                      # our home planet's mean distance from the sun [AU]
P = np.sqrt(d**3/M)                                             # our home planet's revolution period [yr]

N = 20*10**4        # amount of time steps
dt = 20*P/N         # time step

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
            # dropper å gange med m[i] fordi vi antar at planetene ikke trekker på sola?
            g = - G*M/np.linalg.norm(r[i][j])**3*r[i][j]
            v[i+1][j] = v[i][j] + g*dt/2
            r[i+1][j] = r[i][j] + v[i+1][j]*dt
            g = - G*M/np.linalg.norm(r[i+1][j])**3*r[i+1][j]
            v[i+1][j] = v[i+1][j] + g*dt/2
            theta[i+1][j] = theta[i][j] + np.linalg.norm(v[i+1][j])/np.linalg.norm(r[i+1][j])*dt
    return theta, r, v

theta, r, v = simulate_orbits(planets, N, init_angles, r0, v0, M)

for i in range(len(planets)):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planets[i][1], label = planets[i][0])
plt.plot(0, 0, color = 'orange', marker = 'o', label = 'Sun')
plt.legend()
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our planet's orbits around their sun")
plt.show()

#HVA BLIR FEIL HER?
#FIKSE LEGEND?
#SKAL DE TILTE?

'''
Creating legend with loc="best" can be slow with large amounts of data.
fig.canvas.print_figure(bytes_io, **kw)
'''

'''
B2: Comparing your Orbits to Kepler's Laws

Task 1
'''

N_steps = 500           # number of dt we want to retrieve data from 

'''
looking at a dA close to the aphelion
'''
mean_r = 0              # average distance from the star over a small time step
dtheta = 0              # how much the angle between the position and the x-axis has changed over a small time step
                
for i in range(N_steps):
    mean_r += np.linalg.norm(r[i][0])
    dtheta += theta[i][0] - theta[0][0]
mean_r = mean_r/N_steps
dtheta = dtheta/N_steps  

dA = 0.5*mean_r**2*dtheta
print(dA)

'''
looking at a dA close to the perihelion

we know that our planet does 20 revolutions during the simulation period, with
10000 time steps per revolution period. this means that it's about halfway done 
with the first revolution period at around the 5000th time step, and is at
this point close to the perihelion
'''
mean_r = 0              # average distance from the star over a small time step
dtheta = 0              # how much the angle between the position and the x-axis has changed over a small time step
                
for i in range(5000, 5000 + N_steps):
    mean_r += np.linalg.norm(r[i][0])
    dtheta += theta[i][0] - theta[5000][0]
mean_r = mean_r/N_steps
dtheta = dtheta/N_steps  

dA = 0.5*mean_r**2*dtheta
print(dA)

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
        a_dA = 0.5*a_mean_r**2*a_dtheta
        p_dA = 0.5*p_mean_r**2*p_dtheta
        diff = np.abs(a_dA - p_dA)
        rel_err = diff/a_dA
        a_distance = a_mean_r*a_dtheta
        p_distance = p_mean_r*p_dtheta
        print(f'The difference between the area close to the aphelion\nand the area close to the perihelion for {planets[i][0]} is {diff}\nwith a relative error of {rel_err}')
        print(f'{planets[i][0]} travelled {a_distance:.3f} AU while sweeping out the area by the aphelion\nand {p_distance:.3f} AU while sweeping out the area by the perihelion')
        print(f'{planets[i][0]} travelled with a mean velocity of {a_mean_v:.3f} AU/yr while sweeping\nout the area by the aphelion and {p_mean_v:.3f} AU/yr while sweeping out\nthe area by the perihelion')
        
diff_area(planets, N_steps, theta, r, v)


'''
def diff_area(planets, N_steps, dt, r, P):
    for i in range(len(planets)):
        dAa = 0.5*r[0:N_steps, i]**2*2*np.pi/P*dt
        dAp = 0.5*r[5000:5000 + N_steps, i]**2*2*np.pi/P*dt
        
        
        blir denne dthetaen riktig? blir den ikke lik for alle endringer av radius da?
        P er vel heller ikke lik for alle planetene?
        
        
        print(np.sum(dAa))
        print(np.sum(dAp))
        print(f'Differansen mellom de to arealene til planet {i}: {np.abs(np.sum(dAa) - np.sum(dAp))}   |  Den relative feilen: {np.abs(np.sum(dAa)-np.sum(dAp))/np.abs(np.sum(dAp))}')

diff_area(planets, N_steps, dt, r, P)
'''

#m = system.masses                       # each planet's mass [M]
#r = system.radii                        # each planet's radius [km]
#T = system.rotational_periods           # each planet's rotational period [days]

#R = system.star_radius                  # the sun's radius [km]

'''
Task 2
'''
