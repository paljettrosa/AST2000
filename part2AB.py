#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)
plt.rcParams.update({'font.size': 14})

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = 4*np.pi**2                          # gravitation constant [AU**3yr**(-2)M**(-1)]

def plot_orbits(planets, N, a, e, init_angles, a_angles):

    '''
    function for calculating the orbits
    analytically and plotting them
    '''

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
    plt.axis('equal')
    plt.title("Our planets' orbits around their sun")
    plt.tight_layout()


@jit(nopython = True)
def simulate_orbits(planets, N, dt, init_angles, r0, v0, M):

    '''
    function for simulating the orbits
    '''

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




'''
A. Planetary Orbits
'''
'''
Task 1
'''

a = system.semi_major_axes                   # each planet's semi major axis [AU]
e = system.eccentricities                    # each planet's eccentricity
init_angles = system.initial_orbital_angles  # the angle between each planet's initial position and the x-axis
a_angles = system.aphelion_angles            # each planet's initial angle from the aphelion 

N = 1000            

plot_orbits(planets, N, a, e, init_angles, a_angles)


'''
Task 2
'''

G = 4*np.pi**2                           # gravitation constant [AU**3yr**(-2)M**(-1)]
m = system.masses                        # our planet's masses [kg]
M = system.star_mass                     # the sun's mass [M]

init_pos = system.initial_positions      # each planet's initial position [AU]
init_vel = system.initial_velocities     # each planet's initial velocity [AU/yr]

r0 = np.zeros((len(planets), 2))
v0 = np.zeros((len(planets), 2))
for i in range(len(planets)):
    r0[i] = np.array([init_pos[0][i], init_pos[1][i]])
    v0[i] = np.array([init_vel[0][i], init_vel[1][i]])

P = np.sqrt(4*np.pi**2*a[0]**3/(G*(M + m[0])))   # our home planet's revolution period [yr]

N = 40*10**4        # amount of time steps
dt = 40*P/N         # time step

theta, r, v = simulate_orbits(planets, N, dt, init_angles, r0, v0, M)

for i in range(len(planets)):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planets[i][1])
plt.axis('equal')
fig = plt.gcf()
plt.show()
fig.savefig('orbits_analytical&numerical.pdf')




def main():
    '''
    B. Kepler's Laws
    '''
    '''
    Task 1
    '''
    
    def Kepler(planets, N_steps, theta, r, v):

        '''
        function for comparing areas, velocities and distances travelled by
        each planet when sweeping out an area close to its orbit's aphelion
        versus its perihelion
        '''

        n_aphel = np.zeros(len(planets))
        n_perihel = np.zeros(len(planets))
        for i in range(len(planets)):
            distance = np.zeros(len(r))
            distance[0] = np.linalg.norm(r[0, i])
            aphel = distance[0]
            perihel = distance[0]
            for j in range(len(r)-1):
                distance[j+1] = np.linalg.norm(r[j+1, i])
                if distance[j+1] > aphel:
                    aphel = distance[j+1]
                    n_aphel[i] = j+1
                elif distance[j+1] < perihel:
                    perihel = distance[j+1]
                    n_perihel[i] = j+1

        a_mean_r = 0                    
        a_dtheta = 0                    
        a_mean_v = 0                    
        p_mean_r = 0                    
        p_dtheta = 0                    
        p_mean_v = 0
        for i in range(len(planets)):
            for j in range(int(N_steps[i])):
                a_mean_r += np.linalg.norm(r[int(n_aphel[i] + j)][i])
                a_dtheta += theta[int(n_aphel[i] + j)][i] - theta[int(n_aphel[i])][i]
                a_mean_v += np.linalg.norm(v[int(n_aphel[i] + j)][i])
                p_mean_r += np.linalg.norm(r[int(n_perihel[i] + j)][i])
                p_dtheta += theta[int(n_perihel[i] + j)][i] - theta[int(n_perihel[i])][i]
                p_mean_v += np.linalg.norm(v[int(n_perihel[i] + j)][i])

            a_mean_r = a_mean_r/N_steps[i]
            a_dtheta = a_dtheta/N_steps[i]
            a_mean_v = a_mean_v/N_steps[i]
            p_mean_r = p_mean_r/N_steps[i]
            p_dtheta = p_dtheta/N_steps[i]
            p_mean_v = p_mean_v/N_steps[i]
            a_dA = 0.5*a_mean_r**2*a_dtheta
            p_dA = 0.5*p_mean_r**2*p_dtheta
            diff = np.abs(a_dA - p_dA)
            rel_err = diff/a_dA
            a_distance = a_mean_r*a_dtheta
            p_distance = p_mean_r*p_dtheta

            print(f'The difference between the area close to the aphelion\nand the area close to the perihelion for {planets[i][0]} is {diff:.6f} AU^2\nwith a relative error of {rel_err*100:.3f} %')
            print(f'{planets[i][0]} travelled {a_distance:.3f} AU while sweeping out the area by the aphelion\nand {p_distance:.3f} AU while sweeping out the area by the perihelion')
            print(f'{planets[i][0]} travelled with a mean velocity of {a_mean_v:.3f} AU/yr while sweeping\nout the area by the aphelion and {p_mean_v:.3f} AU/yr while sweeping out\nthe area by the perihelion\n')

    N_steps = np.array([500, 600, 6000, 1000, 1500, 800, 3000])    # number of time steps we want to retrieve data from  
                                                                   # for each planet, based off of the size of their orbit         
    Kepler(planets, N_steps, theta, r, v)
    
    
    '''
    Task 2
    '''
    print('\n\n')

    ''' 
    comparing the orbital periods calculated using Kepler's 
    third law, Newton's version, and our simulation results 
    '''
    
    count = [[],[],[],[],[],[],[]]
    Kepler_P = np.sqrt(a**3)
    Newton_P = np.zeros(len(planets))
    numerical_P = np.zeros(len(planets))
    
    for i in range(len(planets)):
        for j in range(1, N):
            if np.sign(r[j][i][1]) != np.sign(r[j-1][i][1]):
                count[i].append(j)
        Newton_P[i] = np.sqrt(4*np.pi**2/(G*(M + m[i]))*a[i]**3)
        numerical_P[i] = (count[i][1] - count[i][0])*dt + (count[i][2] - count[i][1])*dt
        print(f"{planets[i][0]}: numerical approximation: {numerical_P[i]:.3f} years, Kepler's version: {Kepler_P[i]:.3f} years, Newton's version: {Newton_P[i]:.3f} years\n")
    
    
    ''' verifying our simulation results '''
    
    rnew = np.einsum("ijk->kji", r)
    mission.verify_planet_positions(40*P, rnew)

    f = np.load('planet_trajectories.npz')
    times = f['times']
    exact_planet_positions = f['planet_positions']

    system.generate_orbit_video(times, exact_planet_positions, filename = 'orbit_video.xml')


if __name__ == '__main__':
    main()


'''
FROM B:
    
        TASK 1:
    The difference between the area close to the aphelion
    and the area close to the perihelion for Doofenshmirtz is 0.000345 AU^2
    with a relative error of 0.036 %
    Doofenshmirtz travelled 0.543 AU while sweeping out the area by the aphelion
    and 0.555 AU while sweeping out the area by the perihelion
    Doofenshmirtz travelled with a mean velocity of 5.587 AU/yr while sweeping
    out the area by the aphelion and 5.710 AU/yr while sweeping out
    the area by the perihelion

    The difference between the area close to the aphelion
    and the area close to the perihelion for Blossom is 0.000258 AU^2
    with a relative error of 0.018 %
    Blossom travelled 0.510 AU while sweeping out the area by the aphelion
    and 0.534 AU while sweeping out the area by the perihelion
    Blossom travelled with a mean velocity of 4.367 AU/yr while sweeping
    out the area by the aphelion and 4.569 AU/yr while sweeping out
    the area by the perihelion

    The difference between the area close to the aphelion
    and the area close to the perihelion for Bubbles is 0.018342 AU^2
    with a relative error of 0.053 %
    Bubbles travelled 2.001 AU while sweeping out the area by the aphelion
    and 2.387 AU while sweeping out the area by the perihelion
    Bubbles travelled with a mean velocity of 1.713 AU/yr while sweeping
    out the area by the aphelion and 2.043 AU/yr while sweeping out
    the area by the perihelion

    The difference between the area close to the aphelion
    and the area close to the perihelion for Buttercup is 0.001416 AU^2
    with a relative error of 0.040 %
    Buttercup travelled 0.601 AU while sweeping out the area by the aphelion
    and 0.614 AU while sweeping out the area by the perihelion
    Buttercup travelled with a mean velocity of 3.076 AU/yr while sweeping
    out the area by the aphelion and 3.144 AU/yr while sweeping out
    the area by the perihelion

    The difference between the area close to the aphelion
    and the area close to the perihelion for Flora is 0.000511 AU^2
    with a relative error of 0.008 %
    Flora travelled 0.712 AU while sweeping out the area by the aphelion
    and 0.789 AU while sweeping out the area by the perihelion
    Flora travelled with a mean velocity of 2.438 AU/yr while sweeping
    out the area by the aphelion and 2.701 AU/yr while sweeping out
    the area by the perihelion

    The difference between the area close to the aphelion
    and the area close to the perihelion for Stella is 0.000049 AU^2
    with a relative error of 0.002 %
    Stella travelled 0.594 AU while sweeping out the area by the aphelion
    and 0.626 AU while sweeping out the area by the perihelion
    Stella travelled with a mean velocity of 3.804 AU/yr while sweeping
    out the area by the aphelion and 4.010 AU/yr while sweeping out
    the area by the perihelion

    The difference between the area close to the aphelion
    and the area close to the perihelion for Aisha is 0.002630 AU^2
    with a relative error of 0.018 %
    Aisha travelled 1.241 AU while sweeping out the area by the aphelion
    and 1.363 AU while sweeping out the area by the perihelion
    Aisha travelled with a mean velocity of 2.125 AU/yr while sweeping
    out the area by the aphelion and 2.332 AU/yr while sweeping out
    the area by the perihelion


        TASK 2:
    Doofenshmirtz: numerical approximation: 3.896 years, Kepler's version: 6.554 years, Newton's version: 3.896 years

    Blossom: numerical approximation: 7.928 years, Kepler's version: 13.337 years, Newton's version: 7.928 years

    Bubbles: numerical approximation: 107.390 years, Kepler's version: 180.651 years, Newton's version: 107.345 years

    Buttercup: numerical approximation: 23.379 years, Kepler's version: 39.328 years, Newton's version: 23.379 years

    Flora: numerical approximation: 41.646 years, Kepler's version: 70.056 years, Newton's version: 41.639 years

    Stella: numerical approximation: 11.808 years, Kepler's version: 19.863 years, Newton's version: 11.808 years

    Aisha: numerical approximation: 63.753 years, Kepler's version: 107.245 years, Newton's version: 63.752 years


        FROM VERIFICATION AND GENERATING ORBIT VIDEO:       
    The biggest relative deviation was for planet 0, which drifted 0.0329607 % from its actual position.
    Your planet trajectories were satisfyingly calculated. Well done!
    *** Achievement unlocked: Well-behaved planets! ***
    Exact planet trajectories saved to planet_trajectories.npz.
    Generating orbit video with 713 frames.
    Note that planet/moon rotations and moon velocities are adjusted for smooth animation.
    XML file orbit_video.xml was saved in XMLs/.
    It can be viewed in SSView.
'''

