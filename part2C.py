import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import f_part2C as f

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

G = 4*np.pi**2                                      # gravitation constant [AU**3yr**(-2)M**(-1)]
a = system.semi_major_axes[4]                       # Flora's semi major axis [AU]

m = system.masses[4]                                # Flora's mass [M]
M = system.star_mass                                # the sun's mass [M]

P = np.sqrt(4*np.pi**2*a**3/(G*(M + m)))            # Flora's revolution period [yr]

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

cm_r = np.array([0.0, 0.0])                         # redefining the center of mass position
cm_v = np.array([0.0, 0.0])                         # redifining the center of mass velocity

N = 10*10**4        # amount of time steps
dt = 10*P/N         # time step

r, v, E = f.simulate_orbits(N, dt, m, M, F_r0, F_v0, sun_r0, sun_v0)

planet_sun = np.array([['Flora', 'pink'], ['Sun', 'orange']])

for i in range(2):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planet_sun[i][1], label = planet_sun[i][0])
plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Flora's and our sun's orbit around their center of mass")
plt.show()

plt.plot(r[:, 1, 0], r[:, 1, 1], color = 'orange', label = 'Sun')
plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')
plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our sun's orbit around the center of mass")
plt.show()

'''
the center of mass is in the origin, so we see that both Flora and the sun has
elliptical orbits with the center of mass in the origin
'''

for i in range(0, N, 10**4):
    print(E[i])

mean_E = np.mean(E)
min_E = np.min(E)
max_E = np.max(E)
rel_err = np.abs((max_E - min_E)/mean_E)
print(f'Relative error = {rel_err*100:.2f} %')

'''
-0.008939786501259612
-0.009000311609923671
-0.009066804476111764
-0.00913637976111295
-0.009205570122882062
-0.009270443057384448
-0.009326845304083103
-0.009370778663921558
-0.009398867512325685
-0.009408829457603518
Relative error = 6.83 %

when looking at the relative error, we can see that the total energy of our 
two-body system is conserved relatively well throughout the numerical 
simulation of the bodies' orbits 
'''

'''
C2: The Radial Velocity Curve
'''
'''
Task 1
'''

v_pec = np.array([- 1.5*10**(-3), 0.0])         # center of mass velocity relative to observer (peculiar velocity) [AU]

N = 2*10**4
dt = 2*P/N

'''
we assume that the inclination is 90, which means that our line of sight is 
parallell with the plane in which our orbit it situated
'''

v_sun = v[:, 1]                                 # the velocity of our sun relative to the center of mass [AU/yr]
t, v_real, v_obs = f.radial_velocity_curve(N, dt, v_sun, v_pec)

'''
Task 2
'''

f.velocity_data(t, v_obs)

'''
we now want to calculate the mass of an extrasolar planet using the radial 
velocity method. we assume that the inclination is at 90 degrees, which means
sin(i) = 1. this gives us the smallest possible mass of the planet
'''

#radial velocity

'''
C3: The Light Curve
'''
'''
Task 1
'''

'''
we are now interested in a time period where we need to use hours instead
of years, so we change to SI-prefixes
'''

m = m*const.m_sun
M = M*const.m_sun
F_rad = system.radii[4]*1e3
sun_rad = system.star_radius*1e3
r = utils.AU_to_m(r)
v_sun = utils.AU_pr_yr_to_m_pr_s(v_sun)

t0 = 0
for i in range(N):
    if np.abs(r[i, 0, 0] - r[i, 1, 0]) <= F_rad + sun_rad:       # checking when Flora starts eclipsing our sun
        t0 = i
        break

t, F_obs = f.light_curve(t0, N, m, M, F_rad, sun_rad, v_sun)
f.light_curve_data(t, F_obs)

'''
Task 2 and 3
'''

#light curve from group
    
'''
C4: The Radial Velocity Curve with More Planets
'''
'''
Task 1
'''
'''
changing back to astronomical units
'''

M = M/const.m_sun                                   # our sun's mass [M]
F_m = m/const.m_sun                                 # Flora's mass [M]
D_m = system.masses[0]                              # Doofenshmirtz' mass [M]
B_m = system.masses[2]                              # Bubbles' mass [M]
A_m = system.masses[6]                              # Aisha's mass [M] 

D_r0 = np.array([init_pos[0][0], init_pos[1][0]])   # Doofenshmirtz' initial position [AU]
D_v0 = np.array([init_vel[0][0], init_vel[1][0]])   # Doofenshmirtz' initial velocity [AU/yr]

B_r0 = np.array([init_pos[0][2], init_pos[1][2]])   # Bubbles' initial position [AU]
B_v0 = np.array([init_vel[0][2], init_vel[1][2]])   # Bubbles' initial velocity [AU/yr]

A_r0 = np.array([init_pos[0][6], init_pos[1][6]])   # Aisha's initial position [AU]
A_v0 = np.array([init_vel[0][6], init_vel[1][6]])   # Aisha's initial velocity [AU/yr]

sun_r0 = np.array([0.0, 0.0])                       # our sun's initial position [AU]

v0_list = [D_v0, B_v0, F_v0, A_v0]
m_list = [D_m, B_m, F_m, A_m]
planet_momentum = np.zeros(2)

for i in range(4):
    planet_momentum += v0_list[i]*m_list[i]         # the total linear momentum of the planet's we're studying [M*AU/yr]

sun_v0 = - planet_momentum/M                        # our sun's initial velocity [AU/yr]

v0_list.append(sun_v0)
m_list.append(M)

'''
changing frame of reference: cm_r is the positional vector of the center of
mass, which points from the origin and out. since our sun is located in the
origin of our current frame of reference, the positional vector points from
our sun to the center of mass
'''

r0 = np.array([D_r0, B_r0, F_r0, A_r0, sun_r0])
v0 = np.array(v0_list)
m = np.array(m_list)

cm_r = np.zeros(2)
cm_v = np.zeros(2)

for i in range(5):
    cm_r += m[i]/sum(m)*r0[i]                       # center of mass position relative to the sun
    cm_v += m[i]/sum(m)*v0[i]                       # center of mass velocity relative to the sun

    r0[i] = r0[i] - cm_r                            # changing the initial position so it's relative to the center of mass
    v0[i] = v0[i] - cm_v                            # changing the initial velocity so it's relative to the center of mass

sun_r0 = r0[-1]
sun_v0 = v0[-1]

cm_r = np.zeros(2)                                  # placing the center of mass in the origin
cm_v = np.zeros(2)

planet_m = np.sum(m[:4])                            # the sum of the planets' masses [M]               
planet_cm_r0 = np.zeros(2)
planet_cm_v0 = np.zeros(2)

for i in range(4):
    planet_cm_r0 += m[i]/planet_m*r0[i]             # the planets' center of mass position relative to the system's center of mass
    planet_cm_v0 += m[i]/planet_m*v0[i]             # the planets' center of mass velocity relative to the system's center of mass

N = 10*10**4        # amount of time steps
dt = 10*P/N         # time step

r, v = f.simulate_orbits_2(N, dt, planet_m, M, planet_cm_r0, planet_cm_v0, sun_r0, sun_v0)

planet_sun = np.array([["planets' center of mass", 'cyan'], ['Sun', 'orange']])

for i in range(2):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planet_sun[i][1], label = planet_sun[i][0])
plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our planets' center of mass and our sun's orbit around their common center of mass")
plt.show()

plt.plot(r[:, 1, 0], r[:, 1, 1], color = 'orange', label = 'Sun')
plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')
plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our sun's orbit around the center of mass")
plt.show()

'''
Task 2
'''

v_pec = np.array([- 1.5*10**(-3), 0.0])         # center of mass velocity relative to observer (peculiar velocity) [AU]

N = 3*10**4
dt = 3*P/N

v_sun = v[:, 1]                                 # the velocity of our sun relative to the center of mass [AU/yr]
t, v_real, v_obs = f.radial_velocity_curve(N, dt, v_sun, v_pec)