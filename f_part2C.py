import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

def simulate_orbits(N, dt, m, M, planet_r0, planet_v0, star_r0, star_v0):
    G = 4*np.pi**2                                      # gravitation constant [AU**3yr**(-2)M**(-1)]
    
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
    sigma = 0.2*np.max(abs(v_real))             # the standard deviation
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
        outfile.write(' velocity [AU/yr]         time [yr]  \n')
        for vi, ti in zip(v_obs, t):
            outfile.write(f'{vi:.15f} {ti:.15f} \n')
            

def delta(t, v_obs, vr, P, t0):
    delta = 0
    v_mod = np.zeros(len(t))                            # the modelled radial velocity
    
    for i in range(len(t)):
        v_mod[i] = vr*np.cos(2*np.pi/P*(t[i] - t0))        
        delta += (v_obs[i] - v_mod)**2                  # calculating how much the modelled 
                                                        # velocity curve deviates from the noise
    return delta, v_mod

def least_squares(M, t, v_obs, vr_list, P_list, t0_list):
    G = 4*np.pi**2                            # gravitation constant [AU**3yr**(-2)M**(-1)]
    
    vr = vr_list[0]                           # first guess of the sun's radial velocity [AU/yr]
    P = P_list[0]                             # first guess of the sun's revolution period [yr]
    t0 = t0_list[0]                           # first guess of the sun's first peak in radial 
                                              # velocity during the simulation period [yr]
    guess = delta(t, v_obs, vr, P, t0)[0]
    N = len(vr_list)
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                new_guess = delta(vr_list[i], P_list[j], t0_list[k])[0]
                if new_guess < guess:
                    guess = new_guess          # updating our guess if the new value is a better approximation 
                    vr = vr_list[i]
                    P = P_list[j]
                    t0 = t0_list[k]

    m = M**(2/3)*vr*(P/(2*np.pi*G))**(1/3)     # approximation of the smallest possible mass of 
                                               # the extrasolar planet [M]
    return m, vr, P, t0
            

def vr_from_group(m, M, filename):
    with open(filename, 'r') as infile:                                                         
        v_real = []
        t = []
        for line in infile:
            words = line.split()
            v_real.append(float(words[0]))
            t.append(float(words[1]))
        v_real = np.array(v_real)
        t = np.array(t)

    V = np.mean(v_real)                        # mean of peculiar velocity [AU/yr]
    v_obs = v_real - np.full(len(v_real), V)                    

    plt.plot(t, v_obs, 'k')
    plt.plot(t, v_real, 'r')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [km/s]')
    plt.title('Radial velocity curve')
    plt.show()

    vr_list = np.linspace(0.035, 0.06, 20)                    # BYTT UT DISSE VERDIENE MED VERDIER SOM PASSER UTIFRA GRAF!!!!!!!!!!!!!!!!!!!!!!!!!
    P_list = np.linspace(25, 27, 20)
    t0_list = np.linspace(19, 21.5, 20)
    
    est_m, vr, P, t0 = least_squares(M, t, v_obs, vr_list, P_list, t0_list)

    v_mod = delta(t, v_obs, vr, P, t0)[1]
    plt.plot(t, v_obs, 'k--')
    plt.plot(t, v_mod, 'wo')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [km/s]')
    plt.title('Radial velocity curve')
    plt.show()

    print('Parameters are:')
    print('-'*50)
    print(f'    v_star = {vr:10.5f} km/s')
    print(f'         P = {P:10.5f} yr')
    print(f'        t0 = {t0:10.5f} yr')
    print('-'*50)
    print(f'The estimated mass of the planet is {est_m:.8f} solar masses, while the actual mass is {m:.8f} solar masses.')
    print(f'The relative error is {abs(est_m - m)/m*100:.2f} %.')
    print(f'Their inclination could therefore be i = {np.arcsin(est_m/m)/np.pi*180:.2f} deg')
    print('-'*50)
    
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
    # riktig? hvorfor r**3? ikke np.pi*r**2/(np.pi*R**2) = r**2/R**2 siden dette er arealet?
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
    est_m = est_m*const.m_sun                       # estimated smallest mass of the group's planet [kg]
    vr = utils.AU_pr_yr_to_m_pr_s(vr)               # the estimated radial velocity of the group's star [m/s]

    plt.plot(t, F_obs, "p")
    plt.xlabel("time of observation [h]")
    plt.ylabel("relative flux")
    plt.title("Light curve of data recieved from Tiril and Andreas")
    plt.show()
    
    M = M*const.m_sun
    est_radius, est_density = radius_density(est_m, M, vr, t1 - t0)

    print('-'*50)
    print(f'The radius and density of the planet eclipsing the star is respectively {est_radius*1e-3:.2f} km and {est_density:.2f} kg/m^3')
    print(f'Relative errors: {abs(est_radius - radius)/radius*100:.2f} %, {abs(est_density - density)/density*100:.2f} %')
    print('-'*50)
    '''
    calc_density = 3*m*const.sun_m/(4*np.pi*radius**3)
    print(f'Density given: {density} kg/m^3')
    print(f'The calculated density with correct mass and radius is {calc_density}, which is {abs(calc_density - density)/density*100:.2f}')
    print('-'*50)
    '''
    

def simulate_orbits_2(N, dt, planet_m, M, planet_cm_r0, planet_cm_v0, star_r0, star_v0):
    G = 4*np.pi**2                                  # gravitation constant [AU**3yr**(-2)M**(-1)]
    
    r = np.zeros((N, 2, 2))
    v = np.zeros((N, 2, 2))
    #E = np.zeros(N)
    
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