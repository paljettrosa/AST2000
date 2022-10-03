#EGEN KODE
from libraries import *

m_sun = const.m_sun                                 # solar mass [kg]
G = 4*np.pi**2                                      # gravitation constant in AU [AU**3yr**(-2)M**(-1)]

#@jit(nopython = True)
def twobody_orbits(N, dt, m, M, planet_r0, planet_v0, star_r0, star_v0):
    r = np.zeros((N, 2, 2))
    v = np.zeros((N, 2, 2))
    E = np.zeros(N)
    
    r[0] = np.array([planet_r0, star_r0])
    v[0] = np.array([planet_v0, star_v0])
    
    my = m*M/(m + M)                                    # the reduced mass of our system [M]
    R = np.linalg.norm(r[0, 0] - r[0, 1])               # the distance between the two bodies in our system [AU]
    planet_E0 = 0.5*my*np.linalg.norm(planet_v0)**2 - G*(m + M)*my/R
    star_E0 = 0.5*my*np.linalg.norm(star_v0)**2 - G*(m + M)*my/R
    E[0] = planet_E0 + star_E0
    
    for i in range(N - 1):
        R = r[i, 0] - r[i, 1]
        g_planet = - G*M/np.linalg.norm(R)**3*R
        g_star = - G*m/np.linalg.norm(R)**3*(-R)
        
        v[i+1, 0] = v[i, 0] + g_planet*dt/2
        v[i+1, 1] = v[i, 1] + g_star*dt/2
        
        r[i+1, 0] = r[i, 0] + v[i+1, 0]*dt
        r[i+1, 1] = r[i, 1] + v[i+1, 1]*dt
        
        R = r[i+1, 0] - r[i+1, 1]
        g_planet = - G*M/np.linalg.norm(R)**3*R
        g_star = - G*m/np.linalg.norm(R)**3*(-R)
        
        v[i+1, 0] = v[i+1, 0] + g_planet*dt/2
        v[i+1, 1] = v[i+1, 1] + g_star*dt/2
        
        planet_E = 0.5*my*np.linalg.norm(v[i+1, 0])**2 - G*(m + M)*my/np.linalg.norm(R)
        star_E = 0.5*my*np.linalg.norm(v[i+1, 1])**2 - G*(m + M)*my/np.linalg.norm(R)
        E[i+1] = planet_E + star_E
    return r, v, E


def radial_velocity_curve(N, dt, v_star, v_pec):
    t = np.linspace(0, N*dt, N)

    V = np.full(N, v_pec[0])                    # the radial component of the peculiar velocity [AU/yr]
    v_real = v_star[:N, 0] + V                  # the star's true radial velocity [AU/yr]

    plt.plot(t, v_real, color = 'orange', label = 'Sun')
    plt.plot(t, V, color = 'pink', label = 'Peculiar')
    plt.title("Our sun's radial velocity relative to the center of mass,\nand the peculiar velocity of our system")
    plt.xlabel('time [yr]')
    plt.ylabel('velocity [AU/yr]')
    plt.legend()
    plt.show()
    
    ''' calculating noise '''
    
    my = 0.0                                    # the mean noise
    sigma = 0.1*np.max(abs(v_real))             # the standard deviation
    noise = np.random.normal(my, sigma, size = (int(N)))
    v_obs = v_real + noise                      # the observed radial velocity [AU/yr]
    
    plt.plot(t, v_obs, 'k:')
    plt.plot(t, v_real, 'r')
    plt.title('The radial velocity curve of our sun with noise')
    plt.xlabel('time [yr]')
    plt.ylabel('velocity [AU/yr]')
    plt.show()
    
    return t, v_real, v_obs


def velocity_data(t, v_obs):
    with open('velocitydata.txt', 'w') as outfile:
        outfile.write(' time [yr]             velocity [AU/yr] \n')
        for ti, vi in zip(t, v_obs):
            outfile.write(f'{ti:.15f} {vi:.15f} \n')
            
            
@jit(nopython = True)
def delta(t, v_obsreal, vr, P, t0):
    delta = 0
    v_mod = np.zeros(len(t))                            # the modelled radial velocity
    
    for i in range(len(t)):
        v_mod[i] = vr*np.cos(2*np.pi/P*(t[i] - t0))        
        delta = delta + (v_obsreal[i] - v_mod[i])**2    # calculating how much the modelled 
                                                        # velocity curve deviates from the noise                                                
    return delta, v_mod


@jit(nopython = True)
def least_squares(M, t, v_obsreal, vr_list, P_list, t0_list):
    vr = vr_list[0]                           # first guess of the sun's radial velocity [AU/yr]
    P = P_list[0]                             # first guess of the sun's revolution period around the center of mass [yr]
    t0 = t0_list[0]                           # first guess of the sun's first peak in radial 
                                              # velocity during the simulation period [yr]
    guess = delta(t, v_obsreal, vr, P, t0)[0]
    N = len(vr_list)
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                new_guess = delta(t, v_obsreal, vr_list[i], P_list[j], t0_list[k])[0]
                if new_guess < guess:
                    guess = new_guess          # updating our guess if the new value is a better approximation 
                    vr = vr_list[i]
                    P = P_list[j]
                    t0 = t0_list[k]

    m = M**(2/3)*abs(vr)*(P/(2*np.pi*G))**(1/3)     # approximation of the smallest possible mass of 
                                                    # the extrasolar planet [M]
    return m, vr, P, t0     


def vr_from_group(m, M, filename):
    with open(filename, 'r') as infile: 
        t = []                                                        
        v_obs = []
        infile.readline()
        for line in infile:
            words = line.split()
            t.append(float(words[0]))
            v_obs.append(float(words[1]))
        v_obs = np.array(v_obs)
        t = np.array(t)

    V = np.mean(v_obs)                        # mean of peculiar velocity [AU/yr]
    V = np.full(len(v_obs), V)
    v_obsreal = v_obs - V                 

    plt.plot(t, v_obs, 'k', label = 'radial velocity with peculiar velocity')
    plt.plot(t, V, 'y', label = 'peculiar velocity')
    plt.legend()
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [AU/yr]')
    plt.title('Radial velocity curve from data\nrecieved from Oskar and Jannik')
    plt.show()
    
    plt.plot(t, v_obsreal, 'k')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [AU/yr]')
    plt.title('Radial velocity curve from data\nrecieved from Oskar and Jannik\nwithout peculiar velocity')
    plt.show()


    #OSS, METODEN VIRKER IKKE
    
    vr_mean = np.mean(v_obsreal)
    '''
    vr_list = np.linspace(vr_mean - 0.005, vr_mean + 0.005, 50)
    
    tol = 1e-3
    dt = t[1]
    for i in range(300, len(t)):
        if abs(np.sum(abs(v_obsreal[i-100:i+100])) - abs(np.sum(v_obsreal[:200]))) < tol:
            P_list = np.linspace(i*dt-3, i*dt+3, 50)
            if i*dt > len(t)*dt/2:              
                break
            
    tol = 1e-3      
    for i in range(300, len(t)-100):
        if np.sum(v_obsreal[i-100:i+100]) > np.sum(v_obsreal[i-300:i-100]):
            t0_list = np.linspace(t[i-100], t[i+100], 50)
            if abs(abs(np.sum(v_obsreal[i-100:i+100])) - abs(np.sum(v_obsreal[:200]))) < tol:
                print(i*dt)
                break
    
    vr_list = np.linspace(-0.001, 0.001, 50)  
    P_list = np.linspace(37, 43, 50)
    t0_list = np.linspace(6, 11, 50)
    '''
    
    #OSKAR OG JANNIK
    
    vr_list = np.linspace(vr_mean - 1e-6, vr_mean + 1e-6, 100) 
    P_list = np.linspace(34, 40, 100)
    t0_list = np.linspace(11, 17, 100)
    
    
    est_m, vr, P, t0 = least_squares(M, t, v_obsreal, vr_list, P_list, t0_list)
    v_mod = delta(t, v_obsreal, vr, P, t0)[1]
    
    plt.plot(t, v_obsreal, 'k--')
    plt.plot(t, v_mod, 'r')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [AU/yr]')
    plt.title('Modelled radial velocity curve')
    plt.show()

    print(f"The estimated radial velocity of Oskar and Jannik's star is {utils.AU_pr_yr_to_m_pr_s(vr):.5f} m/s")
    print(f"The estimated revolution period for Oskar and Jannik's star is {P:.5f} years")
    print(f"The estimated time stamp of the sun's first peak in radial velocity is at t0 = {t0:.5f} years")
    print(f'The estimated mass of the planet is {est_m:.8f} solar masses, while the actual mass is {m:.8f} solar masses.')
    print(f'The relative error is {abs(est_m - m)/m*100:.2f} %.')
    print(f'Their estimated inclination is i = {np.arcsin(est_m/m)/np.pi*180:.2f} deg')

    return est_m, vr


def light_curve(t0, N, m, M, planet_rad, star_rad, v_star):
    v_star = np.mean(np.linalg.norm(v_star))
    t1 = 2*planet_rad/(v_star*(M/m)) + t0                    # time stamp where the planet's completely eclipsing the star [s]
    t2 = (2*star_rad - 2*planet_rad)/(v_star*(M/m)) + t1     # time stamp where the flux increases again [s]
    t3 = 2*planet_rad/(v_star*(M/m)) + t2                    # time stamp where the planet's no longer eclipsing the star [s]
    
    interval = t3 - t0
    t = np.linspace(t0 - 0.25*interval, t3 + 0.25*interval, N) 
             
    F = np.zeros(N)
    F0 = 1                                                   # the relative light flux of the star when there's no eclipse
    Fp = 1 - planet_rad**2/(star_rad**2)                     # the relative light flux of the star when the planet's fully eclipsing it
    
    ''' we're approximating the planet as a disc since the inclination is 90'''
    
    for i in range(N):
        if t[i] < t0 or t[i] > t3:
            F[i] = F0
        elif t0 <= t[i] <= t1:
            F[i] = F0 + (t[i] - t0)*(Fp - F0)/(t1 - t0)
        elif t1 < t[i] < t2:
            F[i] = Fp
        elif t2 <= t[i] <= t3:
            F[i] = Fp + (t[i] - t2)*(F0 - Fp)/(t3 - t2)
    
    ''' calculating noise '''
    
    my = 0.0                                # the mean value of the noise
    sigma = 10**(-4)*Fp                     # the standard deviation of the noise values
    noise = np.random.normal(my, sigma, N)
    F_obs = F + noise

    plt.plot(t, F_obs, 'k:')                
    plt.plot(t, F, 'r')
    plt.xlabel('time [hours]')
    plt.ylabel('relative light flux')
    plt.title("The relative light flux of the star while the planet's eclipsing it")
    plt.show()
    
    return t, F_obs


def light_curve_data(t, F_obs):
    with open('lightcurvedata.txt', 'w') as outfile:
       outfile.write(' relative flux         time [hours]  \n')
       for Fi, ti in zip(F_obs, t):
           outfile.write(f'{Fi:.15f} {ti:.15f} \n')                               
    
    
def radius_density(m, M, v_star, delta_t):
    v_planet = v_star*M/m
    radius = 0.5*(v_star + v_planet)*delta_t
    density = 3*m/(4*np.pi*radius**3)
    return radius, density


def light_curve_from_group(m, M, filename1, filename2, radius, density):
    with open(filename1, 'r') as infile:
        F_obs = []
        t = []
        for line in infile:
            words = line.split()
            F_obs.append(float(words[0]))
            t.append(float(words[1]))

    F_obs = np.array(F_obs)
    t = np.array(t)/3600                                # SJEKK HVILKE ENHETER DE BRUKER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    '''
    sec_per_hour = 3600 # s/h

    t0 = 10926 * sec_per_hour; 
    t1 = 10931 * sec_per_hour # s, from observation
    GJØR DISSE OBSERVASJONENE OG SJEKK HVILKE ENHETER VI SKAL BRUKE
    PRØV Å DETECTE T0 OG T1 og definer dette utenfor funksjonen så den heller henter det inn som variabler
    '''

    est_m, vr = vr_from_group(m, M, filename2)
    est_m = est_m*m_sun                             # estimated smallest mass of the group's planet [kg]
    vr = utils.AU_pr_yr_to_m_pr_s(vr)               # the estimated radial velocity of the group's star [m/s]

    plt.plot(t, F_obs, "p")
    plt.xlabel("time of observation [h]")
    plt.ylabel("relative flux")
    plt.title("Light curve from data recieved from Oskar and Jannik")
    plt.show()
    
    M = M*m_sun
    est_radius, est_density = radius_density(est_m, M, vr, t1 - t0)

    print(f'The radius and density of the planet eclipsing the star is respectively {est_radius*1e-3:.2f} km and {est_density:.2f} kg/m^3')
    print(f'Relative errors: {abs(est_radius - radius)/radius*100:.2f} %, {abs(est_density - density)/density*100:.2f} %')
    '''
    calc_density = 3*m*const.sun_m/(4*np.pi*radius**3)
    print(f'Density given: {density} kg/m^3')
    print(f'The calculated density with correct mass and radius is {calc_density}, which is {abs(calc_density - density)/density*100:.2f}')
    print('-'*50)
    '''
    #FIKS DEN SISTE DELEN HER
    

#@jit(nopython = True)
def nbody_orbits(N, dt, planet_m, M, planet_cm_r0, planet_cm_v0, star_r0, star_v0):
    r = np.zeros((N, 2, 2))
    v = np.zeros((N, 2, 2))
    
    r[0] = np.array([planet_cm_r0, star_r0])
    v[0] = np.array([planet_cm_v0, star_v0])
                                      
    for i in range(N - 1):
        R = r[i, 0] - r[i, 1]                        # the distance between the star and the center of mass [AU]
        g_planets = - G*M/np.linalg.norm(R)**3*R
        g_star = - G*planet_m/np.linalg.norm(R)**3*(-R)
        
        v[i+1, 0] = v[i, 0] + g_planets*dt/2
        v[i+1, 1] = v[i, 1] + g_star*dt/2
        
        r[i+1, 0] = r[i, 0] + v[i+1, 0]*dt
        r[i+1, 1] = r[i, 1] + v[i+1, 1]*dt
        
        R = r[i+1, 0] - r[i+1, 1]
        g_planets = - G*M/np.linalg.norm(R)**3*R
        g_star = - G*planet_m/np.linalg.norm(R)**3*(-R)
        
        v[i+1, 0] = v[i+1, 0] + g_planets*dt/2
        v[i+1, 1] = v[i+1, 1] + g_star*dt/2
    return r, v

'''
def nbody_orbits_(N, dt, planet_m, M, planets_r0, planets_v0, star_r0, star_v0):
    r = np.zeros((N, len(planets_r0) + 1, 2))
    v = np.zeros((N, len(planets_r0) + 1, 2))
    
    for i in range(len(planets_r0)):
        r[0, i] = planets_r0[i]
        v[0, i] = planets_v0[i]
    
    r[0, -1] = star_r0
    v[0, -1] = star_v0
    
    R = np.zeros((len(planets_r0), 2))
    g_planets = np.zeros((len(planets_r0), 2))
    for i in range(N - 1):
        for j in range(len(planets_r0)):
            R[j] = r[i, j] - r[i, -1]                            # the distance between the star and planet j [AU]
            g_planets[j] = - G*M/np.linalg.norm(R[j])**3*R[j]
            
            v[i+1, j] = v[i, j] + g_planets[j]*dt/2
            r[i+1, j] = r[i, j] + v[i+1, j]*dt
            
        g_star = - G*planet_m/np.linalg.norm(np.sum(R))**3*(-np.sum(R))
        
        v[i+1, -1] = v[i, -1] + g_star*dt/2
        r[i+1, -1] = r[i, -1] + v[i+1, -1]*dt
        
        for j in range(len(planets_r0)):
            R[j] = r[i+1, j] - r[i+1, -1]
            g_planets[j] = - G*M/np.linalg.norm(R[j])**3*R[j]
            
            v[i+1, j] = v[i+1, j] + g_planets[j]*dt/2
            
        g_star = - G*planet_m/np.linalg.norm(np.sum(R))**3*(-np.sum(R))
        
        v[i+1, -1] = v[i+1, -1] + g_star*dt/2
    return r, v
'''



'''
C1: The Solar Orbit 
'''
'''
we choose Flora, because it's the planet with the second largest mass, and 
it's the fourth planet away from the sun
'''

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

N = 2*10**5         # amount of time steps
dt = 20*P/N         # time step

r, v, E = twobody_orbits(N, dt, m, M, F_r0, F_v0, sun_r0, sun_v0)

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

t = np.linspace(0, N*dt, N)
mean_E = np.mean(E)
mean_E_array = np.full(N, mean_E)

plt.plot(t, E, color = 'darkorchid', label = 'energy of system throughout simulation')
plt.plot(t, mean_E_array, color = 'olivedrab', label = 'mean energy of system')
plt.legend(loc = 'lower left')
plt.xlabel('time [yr]')
plt.ylabel('energy [J]')
plt.title("The total energy of our two-body system\nduring the simulation of their orbits")
plt.show()

min_E = np.min(E)
max_E = np.max(E)
rel_err = np.abs((max_E - min_E)/mean_E)
print(f'Relative error = {rel_err*100:.2f} %')

'''
Relative error = 6.83 %

when looking at the relative error, we can see that the total energy of our 
two-body system is conserved relatively well throughout the numerical 
simulation of the bodies' orbits 
'''
# SPØR OSKAR HVORDAN HANS BLE SÅ LITEN




'''
C2: The Radial Velocity Curve
'''
'''
Task 1
'''

v_pec = np.array([- 1.5*10**(-3), 0.0])         # center of mass velocity relative to observer (peculiar velocity) [AU/yr]

N = 2*10**4
dt = 2*P/N

'''
we assume that the inclination is 90, which means that our line of sight is 
parallell with the plane in which our orbit it situated
'''

v_sun = v[:, 1]                                 # the velocity of our sun relative to the center of mass [AU/yr]
t, v_real, v_obs = radial_velocity_curve(N, dt, v_sun, v_pec)

'''
Task 2
'''

velocity_data(t, v_obs)

'''
we now want to calculate the mass of an extrasolar planet using the radial 
velocity method. we assume that the inclination is at 90 degrees, which means
sin(i) = 1. this gives us the smallest possible mass of the planet
'''

m_janniesc = 1.3966746879342887e-06             # the actual mass of the planet Oskar and Jannik used for their two-body system [M]
M_janniesc = 4.4408144144136115                 # the actual mass of Oskar and Jannik's sun [M]

est_m, vr_janniesc = vr_from_group(m_janniesc, M_janniesc, 'Radial-Velocity.txt')




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

m = m*m_sun
M = M*m_sun
F_rad = system.radii[4]*1e3
sun_rad = system.star_radius*1e3
r = utils.AU_to_m(r)
v_sun = utils.AU_pr_yr_to_m_pr_s(v_sun)

t0 = 0
for i in range(N):
    if np.abs(r[i, 0, 0] - r[i, 1, 0]) <= F_rad + sun_rad:       # checking when Flora starts eclipsing our sun
        t0 = i
        break

t, F_obs = light_curve(t0, N, m, M, F_rad, sun_rad, v_sun)
light_curve_data(t, F_obs)

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

''' changing back to astronomical units '''

M = M/m_sun                                         # our sun's mass [M]
F_m = m/m_sun                                       # Flora's mass [M]
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

N = 2*10**5           # amount of time steps
dt = 20*P/N           # time step

r, v = nbody_orbits(N, dt, planet_m, M, planet_cm_r0, planet_cm_v0, sun_r0, sun_v0)

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
planets_r0 = r0[:4]
planets_v0 = v0[:4]

r, v = nbody_orbits_(N, dt, planet_m, M, planets_r0, planets_v0, sun_r0, sun_v0)

for i in range(5):
    plt.plot(r[:, i, 0], r[:, i, 1])
#plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our planets' and sun's orbit around their common center of mass")
plt.show()

plt.plot(r[:, -1, 0], r[:, -1, 1], color = 'orange', label = 'Sun')
plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')
plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title("Our sun's orbit around the center of mass")
plt.show()
'''

'''
Task 2
'''

v_pec = np.array([- 1.5*10**(-3), 0.0])         # center of mass velocity relative to observer (peculiar velocity) [AU]

N = 3*10**4
dt = 3*P/N

v_sun = v[:, 1]                                 # the velocity of our sun relative to the center of mass [AU/yr]
t, v_real, v_obs = radial_velocity_curve(N, dt, v_sun, v_pec)