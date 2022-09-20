import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from numba import jit

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

'''
C1: The Solar Orbit 
'''

# [M] is the unit for solar mass
'''
we choose Flora, because it's the planet with the second largest mass, and 
it's the fourth planet away from the sun
'''

a = system.semi_major_axes[4]                       # Flora's semi major axis [AU]
e = system.eccentricities[4]                        # Flora's eccentricity
init_angle = system.initial_orbital_angles[4]       # the angle between Flora's initial position and the x-axis
a_angle = system.aphelion_angles[4]                 # Flora's initial angle from the aphelion 

m = system.masses[4]                                # Flora's mass [M]
M = system.star_mass                                # the sun's mass [M]

perihelion = a*(1 - e**2)/(1 + e*np.cos(0))         # the perihelion of Flora's orbit around the sun [AU]
aphelion = a*(1 - e**2)/(1 + e*np.cos(a_angle))     # the aphelion of Flora's orbit around the sun [AU]                     
d = (perihelion + aphelion)/2                       # Flora's mean distance from the sun [AU]
P = np.sqrt(d**3/M)                                 # Flora's revolution period around the sun [yr]

init_pos = system.initial_positions
init_vel = system.initial_velocities

F_r0 = np.array([init_pos[0][4], init_pos[1][4]])   # Flora's initial position [AU]
F_v0 = np.array([init_vel[0][4], init_vel[1][4]])   # Flora's initial velocity [AU/yr]

sun_r0 = np.array([0.0, 0.0])                       # our sun's initial position [AU]
sun_v0 = - F_v0*m/M                                 # our sun's initial velocity [AU/yr]

cm_r = M/(m + M)*sun_r0 + m/(m + M)*F_r0            # center of mass position relative to the sun
cm_v = M/(m + M)*sun_v0 + m/(m + M)*F_v0            # center of mass velocity relative to the sun

'''
changing frame of reference: cm_r is the positional vector of the center of
mass, which points from the origin and out. since our sun is located in the
origin of our current frame of reference, the positional vector points from
our sun to the center of mass
'''

F_r0 = F_r0 - cm_r                                  # Flora's initial position relative to the center of mass
F_v0 = F_v0 - cm_v                                  # Flora's initial velocity relative to the center of mass

sun_r0 = sun_r0 - cm_r                              # our sun's initial position relative to the center of mass
sun_v0 = sun_v0 - cm_v                              # our sun's initial velocity relative to the center of mass

#cm_r = np.array([0.0, 0.0])                         # center of mass position
#cm_v = np.array([0.0, 0.0])                         # center of mass velocity

N = 10*10**4        # amount of time steps
dt = 10*P/N         # time step

#@jit(nopython = True)
def simulate_orbits(N, dt, m, M, planet_r0, planet_v0, star_r0, star_v0):
    G = 4*np.pi**2                          # gravitation constant [AU**3yr**(-2)M**(-1)]
    r = np.zeros((N, 2, 2))
    v = np.zeros((N, 2, 2))
    r[0] = np.array([planet_r0, star_r0])
    v[0] = np.array([planet_v0, star_v0])
    for i in range(N - 1):
        R = r[i, 0] - r[i, 1]
        g_planet = - G*M/np.linalg.norm(R)**3*R
        g_star = - G*m/np.linalg.norm(R)**3*(- R)
        
        v[i+1, 0] = v[i, 0] + g_planet*dt/2
        v[i+1, 1] = v[i, 1] + g_star*dt/2
        
        r[i+1, 0] = r[i, 0] + v[i+1, 0]*dt
        r[i+1, 1] = r[i, 1] + v[i+1, 1]*dt
        
        R = r[i+1, 0] - r[i+1, 1]
        g_planet = - G*M/np.linalg.norm(R)**3*R
        g_star = - G*m/np.linalg.norm(R)**3*(- R)
        
        v[i+1, 0] = v[i+1, 0] + g_planet*dt/2
        v[i+1, 1] = v[i+1, 1] + g_star*dt/2
    return r, v

r, v = simulate_orbits(N, dt, m, M, F_r0, F_v0, sun_r0, sun_v0)

planet_sun = np.array([['Flora', 'pink'], ['Sun', 'orange']])

for i in range(2):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planet_sun[i][1], label = planet_sun[i][0])
#plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')
plt.legend()
plt.title("Flora's and our sun's orbit around their center of mass")
plt.show()

plt.plot(r[:, 1, 0], r[:, 1, 1], color = 'orange', label = 'Sun')
#plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')
plt.legend()
plt.title("Our sun's orbit around the center of mass")
plt.show()

'''
the center of mass is in the origin, so we see that both Flora and the sun has
elliptical orbits bith the center of mass in one of the focal points
'''
