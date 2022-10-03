#EGEN KODE
from libraries import *

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

@jit(nopython = True)
def simulate_orbits(planets, N, dt, init_angles, r0, v0, M):
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




'''
A. Planetary Orbits
'''
'''
Task 1
'''

# [M] is the unit for solar mass

a = system.semi_major_axes                   # each planet's semi major axis [AU]
e = system.eccentricities                    # each planet's eccentricity
init_angles = system.initial_orbital_angles  # the angle between each planet's initial position and the x-axis
a_angles = system.aphelion_angles            # each planet's initial angle from the aphelion 

N = 1000            #amount of time steps

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
plt.show()




def main():
    '''
    B. Kepler's Laws
    '''
    '''
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
            print(f'The difference between the area close to the aphelion\nand the area close to the perihelion for {planets[i][0]} is {diff:.2f} km^2\nwith a relative error of {rel_err*100:.2f} %')
            print(f'{planets[i][0]} travelled {a_distance:.3f} AU while sweeping out the area by the aphelion\nand {p_distance:.3f} AU while sweeping out the area by the perihelion')
            print(f'{planets[i][0]} travelled with a mean velocity of {a_mean_v:.3f} AU/yr while sweeping\nout the area by the aphelion and {p_mean_v:.3f} AU/yr while sweeping out\nthe area by the perihelion\n')
            
    diff_area(planets, N_steps, theta, r, v)
    
    
    '''
    Task 2
    '''
    print('\n\n')
    
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
    
    r_reshaped = np.reshape(r, (2, 7, 400000))
 
    mission.verify_planet_positions(40*P, r_reshaped)


'''changing variable names for part 3'''

dt_p = dt
r_p = r
v_p = v


if __name__ == '__main__':
    main()


'''
FROM B:
    
    TASK 1:
        
The difference between the area close to the aphelion
and the area close to the perihelion for Doofenshmirtz is 6064006790944.00 km^2
with a relative error of 0.03 %
Doofenshmirtz travelled 0.534 AU while sweeping out the area by the aphelion
and 0.546 AU while sweeping out the area by the perihelion
Doofenshmirtz travelled with a mean velocity of 5.587 AU/yr while sweeping
out the area by the aphelion and 5.710 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Blossom is 36039808072756.00 km^2
with a relative error of 0.13 %
Blossom travelled 0.421 AU while sweeping out the area by the aphelion
and 0.422 AU while sweeping out the area by the perihelion
Blossom travelled with a mean velocity of 4.389 AU/yr while sweeping
out the area by the aphelion and 4.400 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Bubbles is 32905150827816.00 km^2
with a relative error of 0.05 %
Bubbles travelled 0.199 AU while sweeping out the area by the aphelion
and 0.199 AU while sweeping out the area by the perihelion
Bubbles travelled with a mean velocity of 2.047 AU/yr while sweeping
out the area by the aphelion and 2.041 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Buttercup is 3718919652664.00 km^2
with a relative error of 0.01 %
Buttercup travelled 0.300 AU while sweeping out the area by the aphelion
and 0.301 AU while sweeping out the area by the perihelion
Buttercup travelled with a mean velocity of 3.122 AU/yr while sweeping
out the area by the aphelion and 3.137 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Flora is 16666989451104.00 km^2
with a relative error of 0.04 %
Flora travelled 0.240 AU while sweeping out the area by the aphelion
and 0.238 AU while sweeping out the area by the perihelion
Flora travelled with a mean velocity of 2.507 AU/yr while sweeping
out the area by the aphelion and 2.478 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Stella is 11649877147408.00 km^2
with a relative error of 0.04 %
Stella travelled 0.373 AU while sweeping out the area by the aphelion
and 0.382 AU while sweeping out the area by the perihelion
Stella travelled with a mean velocity of 3.888 AU/yr while sweeping
out the area by the aphelion and 3.985 AU/yr while sweeping out
the area by the perihelion

The difference between the area close to the aphelion
and the area close to the perihelion for Aisha is 25014147117552.00 km^2
with a relative error of 0.05 %
Aisha travelled 0.210 AU while sweeping out the area by the aphelion
and 0.209 AU while sweeping out the area by the perihelion
Aisha travelled with a mean velocity of 2.178 AU/yr while sweeping
out the area by the aphelion and 2.163 AU/yr while sweeping out
the area by the perihelion


    TASK 2:
        
Doofenshmirtz: numerical approximation: 3.896 years, Kepler's version: 6.554 years, Newton's version: 3.896 years

Blossom: numerical approximation: 7.750 years, Kepler's version: 13.337 years, Newton's version: 7.928 years

Bubbles: numerical approximation: 117.007 years, Kepler's version: 180.651 years, Newton's version: 107.345 years

Buttercup: numerical approximation: 23.266 years, Kepler's version: 39.328 years, Newton's version: 23.379 years

Flora: numerical approximation: 38.953 years, Kepler's version: 70.056 years, Newton's version: 41.639 years

Stella: numerical approximation: 12.031 years, Kepler's version: 19.863 years, Newton's version: 11.808 years

Aisha: numerical approximation: 61.305 years, Kepler's version: 107.245 years, Newton's version: 63.752 years



FROM VERIFICATION:
    
The biggest relative deviation was for planet 0, which drifted 1108.33 % from its actual position.
Your planets are not where they should be after 20 orbits of your home planet.
Check your program for flaws or experiment with your time step for more precise trajectories.
'''

