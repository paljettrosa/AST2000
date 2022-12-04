#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import ast2000tools.constants as const
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

m_sun = const.m_sun                                 # solar mass [kg]
G = 4*np.pi**2                                      # gravitation constant in AU [AU**3yr**(-2)M**(-1)]
yr = const.yr                                       # one year [s]

def twobody_orbits(N, dt, m, M, planet_r0, planet_v0, star_r0, star_v0):

    '''
    function for simulating two-body system
    '''

    r = np.zeros((N, 2, 2))
    v = np.zeros((N, 2, 2))
    
    r[0] = np.array([planet_r0, star_r0])
    v[0] = np.array([planet_v0, star_v0])
    
    for i in range(N - 1):
        R = r[i, 0] - r[i, 1]                           # the distance between the two bodies in our system [AU]
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
    return r, v


def energy(N, dt, m, M, r, v):

    '''
    function for calculating energy in two-body system
    '''

    t = np.linspace(0, N*dt, N)
    my = m*M/(m + M)                                         # the reduced mass of our system [M]
    R = r[:N, 0] - r[:N, 1]                                  # the distance between the two bodies in our system [AU]
    V = v[:N, 0] - v[:N, 1]                                  # the velocity of the two bodies in our system relative to each other [AU/yr]
    
    E = np.zeros(N)
    for i in range(N):
        E[i] = 0.5*my*np.linalg.norm(V[i])**2 - G*(m + M)*my/np.linalg.norm(R[i])
        
    mean_E = np.mean(E)
    mean_E_array = np.full(N, mean_E)
    
    plt.plot(t, E, color = 'violet', label = 'energy of system throughout simulation')
    plt.plot(t, mean_E_array, color = 'slateblue', label = 'mean energy of system')
    plt.legend(loc = 'lower left')
    plt.xlabel('time [yr]')
    plt.ylabel('energy [J]')
    plt.title(f"The total energy of our two-body system\nduring the first {N*dt:.1f} years of the simulation\n")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('energy_twobody.pdf')

    min_E = np.min(E)
    max_E = np.max(E)
    rel_err = np.abs((max_E - min_E)/mean_E)
    print(f'The relative error of the estimated energy of the system is {rel_err*100:.2e} %')
    
    
def angularmomentum(N, dt, m, M, r, v):

    '''
    function for calculating angular momentum in two-body system
    '''

    t = np.linspace(0, N*dt, N)
    my = my = m*M/(m + M)                                    # the reduced mass of our system [M]
    R = r[:N, 0] - r[:N, 1]                                  # the distance between the two bodies in our system [AU]
    V = v[:N, 0] - v[:N, 1]                                  # the velocity of the two bodies in our system relative to each other [AU/yr]
    
    P = np.zeros(N)
    for i in range(N):
        P[i] = np.linalg.norm(np.cross(R[i], my*V[i]))
        
    mean_P = np.mean(P)
    mean_P_array = np.full(N, mean_P)
    
    plt.plot(t, P, color = 'turquoise', label = 'calculated angular momentum')
    plt.plot(t, mean_P_array, color = 'gold', label = 'mean angular momentum of system')
    plt.legend(loc = 'lower left')
    plt.xlabel('time [yr]')
    plt.ylabel('angular momentum [Ns]')
    plt.title(f"The total angular momentum of our two-body system\nduring the first {N*dt:.1f} years of the simulation\n")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('angularmomentum_twobody.pdf')

    min_P = np.min(P)
    max_P = np.max(P)
    rel_err = np.abs((max_P - min_P)/mean_P)
    print(f'The relative error of the estimated angular momentum of the system is {rel_err*100:.2e} %')


def radial_velocity_curve(N, dt, v_star, v_pec, nbody=False):

    '''
    function for creating and plotting radial velocity curve
    '''

    t = np.linspace(0, N*dt, N)

    V = np.full(N, v_pec[0])                    # the radial component of the peculiar velocity [AU/yr]
    v_real = v_star[:N, 0] + V                  # the star's true radial velocity [AU/yr]

    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_real), color = 'orange', label = 'Sun')
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(V), color = 'pink', label = 'peculiar')
    plt.title("Our sun's radial velocity relative to the cm,\nand the peculiar velocity of our system")
    plt.xlabel('time [yr]')
    plt.ylabel('velocity [m/s]')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if nbody == False:
        fig.savefig('sun_vr&pec.pdf')
    else:
        fig.savefig('sun_vr&pec_nbody.pdf')
    
    ''' calculating noise '''
    
    my = 0.0                                    # the mean noise
    sigma = 0.1*np.max(abs(v_real))             # the standard deviation
    noise = np.random.normal(my, sigma, size = (int(N)))
    v_obs = v_real + noise                      # the observed radial velocity [AU/yr]
    
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obs), ':', color = 'plum')
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_real), color = 'purple')
    plt.title('The radial velocity curve of our sun with noise')
    plt.xlabel('time [yr]')
    plt.ylabel('velocity [m/s]')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if nbody == False:
        fig.savefig('sun_vr_wnoise.pdf')
    else:
        fig.savefig('sun_vr_wnoise_nbody.pdf')
    
    return t, v_real, v_obs


def velocity_data(t, v_obs, filename):

    '''
    function for creating textfile containing radial velocity data
    '''
    
    with open(filename, 'w') as outfile:
        outfile.write(' time [yr]           velocity [AU/yr] \n')
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

    '''
    function for modelling radial velocity curve by 
    analyzing data from group's two-body system
    '''
    
    with open(filename, 'r') as infile: 
        t = []                                                        
        v_obs = []
        infile.readline()
        for line in infile:
            words = line.split()
            t.append(float(words[0]))
            v_obs.append(float(words[1]))
        t = np.array(t)
        v_obs = np.array(v_obs)

    V = np.mean(v_obs)                        # mean of peculiar velocity [AU/yr]

    print(f"The estimated peculiar velocity of the group's system is {V:.3g} AU/yr")

    V = np.full(len(v_obs), V)
    v_obsreal = v_obs - V    
           
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obs), ':', color = 'skyblue', label = 'radial velocity with peculiar velocity')
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(V), color = 'royalblue', label = 'peculiar velocity')
    plt.legend()
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [m/s]')
    plt.title('Radial velocity curve made with data\nrecieved from group')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('vrcurve_wpec_group.pdf')
    
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obsreal), ':', color = 'skyblue')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [m/s]')
    plt.title('Radial velocity curve made with data recieved\nfrom group without peculiar velocity')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('vrcurve_wopec_group.pdf')
    
    ''' approximation values gathered from plots '''
    
    vr_mean = np.mean(v_obsreal)
    vr_list = np.linspace(vr_mean - 1e-6, vr_mean + 1e-6, 50) 
    P_list = np.linspace(32, 38, 50)
    t0_list = np.linspace(14, 20, 50)
    
    est_m, vr, P, t0 = least_squares(M, t, v_obsreal, vr_list, P_list, t0_list)
    v_mod = delta(t, v_obsreal, vr, P, t0)[1]

    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obsreal), ':', color = 'skyblue')
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_mod), color = 'royalblue')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [m/s]')
    plt.title('Modelled radial velocity curve')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('modelled_vrcurve_group.pdf')

    print(f"The estimated radial velocity of the group's star is {utils.AU_pr_yr_to_m_pr_s(vr):.3f} m/s")
    print(f"The estimated revolution period for the group's star is {P:.3f} years")
    print(f"The estimated time stamp of the sun's first peak in radial velocity is at t0 = {t0:.3f} years")
    print(f'The estimated mass of the planet is {est_m:.3e} solar masses, while the actual mass is {m:.3e} solar masses.')
    print(f'The relative error is {abs(est_m - m)/m*100:.2f} %.')
    return est_m, vr


def light_curve(t0, N, m, M, planet_rad, star_rad, v_star):

    '''
    function for creating and plotting light curve
    '''

    v_star = np.mean(np.linalg.norm(v_star, axis=1))
    t1 = 2*planet_rad/(v_star*(M/m))/3600 + t0                    # time stamp where the planet's completely eclipsing the star [hours]
    t2 = (2*star_rad - 2*planet_rad)/(v_star*(M/m))/3600 + t1     # time stamp where the flux increases again [hours]
    t3 = 2*planet_rad/(v_star*(M/m))/3600 + t2                    # time stamp where the planet's no longer eclipsing the star [hours]
    
    interval = t3 - t0
    t = np.linspace(t0 - 0.25*interval, t3 + 0.25*interval, N) 
             
    F = np.zeros(N)
    F0 = 1                                                        # the relative light flux of the star when there's no eclipse
    Fp = 1 - planet_rad**2/(star_rad**2)                          # the relative light flux of the star when the planet's fully eclipsing it

    print(f"Flora uses approximately {t1-t0:.2f} hours to fully eclipse the sun")
    print(f"The relative light flux when Flora is fully eclipsing the sun is {Fp:.4f}")
    
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

    plt.plot(t, F_obs, ':', color = 'navajowhite')                
    plt.plot(t, F, color = 'orange')
    plt.xlabel('time of observation [hours]')
    plt.ylabel('relative light flux')
    plt.title("The relative light flux of the star while the\nplanet's eclipsing it")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('lightcurve.pdf')
    
    return t, F_obs


def light_curve_data(t, F_obs, filename):

    '''
    function for creating textfile containing light curve data
    '''
    
    with open(filename, 'w') as outfile:
       outfile.write(' time [hours]         relative flux  \n')
       for ti, Fi in zip(t, F_obs):
           outfile.write(f'{ti:.15f} {Fi:.15f} \n')                               
    
    
def radius_density(m, M, v_star, delta_t):

    '''
    function for calculating the radius and
    density of the group's planet
    '''
    
    v_planet = v_star*M/m   
    radius = abs(0.5*(v_star + v_planet)*delta_t)
    density = 3*m/(4*np.pi*radius**3)
    return radius, density


def light_curve_from_group(M, radius, density, est_m, vr, filename_lc):

    '''
    function for modelling light curve and 
    analyzing data from group
    '''
    
    with open(filename_lc, 'r') as infile:
        t = []
        F_obs = []
        infile.readline()
        for line in infile:
            words = line.split()
            t.append(float(words[0]))
            F_obs.append(float(words[1]))

    t = np.array(t)
    F_obs = np.array(F_obs)
    F_approx = np.zeros_like(F_obs)

    for i in range(0, len(F_obs)+1, 1000):
        if i == 0:
            F_approx[:i+500] = np.mean(F_obs[:i+500])
        elif i == len(F_obs):
            F_approx[i-500:] = np.mean(F_obs[i-500:])
        else:
            F_approx[i-500:i+500] = np.mean(F_obs[i-500:i+500])
    
    plt.plot(t, F_obs, ':', color = 'pink', label = 'light curve with noise')
    plt.plot(t, F_approx, color = 'hotpink', label = 'approximated light curve')
    plt.xlabel("time of observation [hours]")
    plt.ylabel("relative flux")
    plt.title("Light curve made with data recieved\nfrom group")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('lightcurve_group.pdf')
    
    ''' calculating when the eclipse is starting and when the planet's fully eclipsing the sun '''

    dt = t[1] - t[0] 
    
    i = 0
    while F_obs[i] >= np.min(F_obs[:int(len(t)/8)]):
        i += 1
    t0 = i*dt + t[0]
    
    while F_obs[i] >= np.max(F_obs[int(len(t)/2)-100:int(len(t)/2)+100]):
        i += 1
    t1 = i*dt + t[0]

    delta_t = (t1 - t0)*3600                        # the time the planet uses to fully eclipse the sun [s] 
    est_radius, est_density = radius_density(est_m, M, vr, delta_t)
    print(f'The estimated radius of the planet eclipsing the star is {est_radius*1e-3:.2f} km, while the actual radius is {radius*1e-3:.2f} km. The relative error is {abs(est_radius - radius)/radius*100:.2f} %')
    print(f'The estimated density of the planet eclipsing the star is {est_density:.2f} kg/m^3, while the actual density is {density:.2f} kg/m^3. The relative error is {abs(est_density - density)/density*100:.2f} %') 


def nbody_orbits(N, dt, planets_m, M, planets_r0, planets_v0, planet_cm_r, planet_cm_v, star_r0, star_v0):

    '''
    function for simulating n-body system
    '''
    
    r = np.zeros((N, len(planets_r0) + 1, 2))
    v = np.zeros((N, len(planets_r0) + 1, 2))
    
    for i in range(len(planets_r0)):
        r[0, i] = planets_r0[i]
        v[0, i] = planets_v0[i]
    
    r[0, -1] = star_r0
    v[0, -1] = star_v0

    for i in range(N - 1):
        for j in range(len(planets_r0)):
            R = r[i, j] - r[i, -1]                                   # planet j's positional vector relative to the star [AU]
            g_planet = - G*M/np.linalg.norm(R)**3*R
            
            v[i+1, j] = v[i, j] + g_planet*dt/2
            r[i+1, j] = r[i, j] + v[i+1, j]*dt

        R = planet_cm_r - r[i, -1]                                   # the planets' center of mass positional vector relative to the star [AU] 

        g_planet_cm = - G*M/np.linalg.norm(R)**3*R
        planet_cm_v = planet_cm_v + g_planet_cm*dt/2
        planet_cm_r = planet_cm_r + planet_cm_v*dt

        g_star = - G*sum(planets_m)/np.linalg.norm(R)**3*(-R)
        v[i+1, -1] = v[i, -1] + g_star*dt/2
        r[i+1, -1] = r[i, -1] + v[i+1, -1]*dt

        for j in range(len(planets_r0)):
            R = r[i+1, j] - r[i+1, -1]
            g_planet = - G*M/np.linalg.norm(R)**3*R

            v[i+1, j] = v[i+1, j] + g_planet*dt/2

        R = planet_cm_r - r[i+1, -1]                                 # the planets' center of mass positional vector relative to the star [AU] 
        planet_cm_v = planet_cm_v + g_planet_cm*dt/2

        g_star = - G*sum(planets_m)/np.linalg.norm(R)**3*(-R)
        v[i+1, -1] = v[i+1, -1] + g_star*dt/2

    return r, v


def vr_from_group_nbody(no, M, mean_momentum_twobody, filename):

    '''
    function for modelling radial velocity curve by 
    analyzing data from group's n-body system
    '''
    
    with open(filename, 'r') as infile: 
        t = []                                                        
        v_obs = []
        infile.readline()
        for line in infile:
            words = line.split()
            t.append(float(words[0]))
            v_obs.append(float(words[1]))
        t = np.array(t)
        v_obs = np.array(v_obs)

    V = np.mean(v_obs)                        # mean of peculiar velocity [AU/yr]

    print(f"The estimated peculiar velocity of the group's system is {V:.3g} AU/yr")

    V = np.full(len(v_obs), V)
    v_obsreal = v_obs - V    
           
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obs), ':', color = 'skyblue', label = 'radial velocity with peculiar velocity')
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(V), color = 'royalblue', label = 'peculiar velocity')
    plt.legend()
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [m/s]')
    plt.title('Radial velocity curve made with data\nrecieved from group')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('vrcurve_wpec_group_nbody.pdf')
    
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obsreal), ':', color = 'skyblue')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [m/s]')
    plt.title('Radial velocity curve made with data recieved\nfrom group without peculiar velocity')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('vrcurve_wopec_group_nbody.pdf')
    
    ''' approximation values gathered from plots '''
    
    vr_mean = np.mean(v_obsreal)
    vr_list = np.linspace(vr_mean - 0.0002, vr_mean + 0.0002, 50) 
    P_list = np.linspace(162, 168, 50)
    t0_list = np.linspace(45, 51, 50)
    
    est_m, vr, P, t0 = least_squares(M, t, v_obsreal, vr_list, P_list, t0_list)
    v_mod = delta(t, v_obsreal, vr, P, t0)[1]

    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_obsreal), ':', color = 'skyblue')
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(v_mod), color = 'royalblue')
    plt.xlabel('time [yr]')
    plt.ylabel('radial velocity [m/s]')
    plt.title('Modelled radial velocity curve')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('modelled_vrcurve_group_nbody.pdf')

    mean_momentum = vr*M
    est_no = mean_momentum/mean_momentum_twobody

    print(f"The estimated radial velocity of the group's star is {utils.AU_pr_yr_to_m_pr_s(vr):.3f} m/s")
    print(f"The estimated revolution period for the group's star is {P:.3f} years")
    print(f"The estimated time stamp of the sun's first peak in radial velocity is at t0 = {t0:.3f} years")
    print(f'The estimated number of planets is {est_no:.2f}, while the actual number of planets is {no}.')
    return est_no, vr




'''
C. Can Extraterrestrials Discover the Planets in your Solar System?
'''
'''
C1. The Solar Orbit 
'''
'''
we choose Flora, because it's the planet with the second largest mass, and 
it's not too far away from the sun
'''

a = system.semi_major_axes[4]                       # Flora's semi major axis [AU]

m = system.masses[4]                                # Flora's mass [M]
M = system.star_mass                                # the sun's mass [M]

P = np.sqrt(4*np.pi**2*a**3/(G*(M + m)))            # Flora's revolution period [yr]

init_pos = system.initial_positions
init_vel = system.initial_velocities

F_r0 = np.array([init_pos[0, 4], init_pos[1, 4]])   # Flora's initial position [AU]
F_v0 = np.array([init_vel[0, 4], init_vel[1, 4]])   # Flora's initial velocity [AU/yr]

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

N = 40*10**4        # amount of time steps
dt = 20*P/N         # time step

r, v = twobody_orbits(N, dt, m, M, F_r0, F_v0, sun_r0, sun_v0)

planet_sun = np.array([['Flora', 'pink'], ['Sun', 'orange']])

for i in range(2):
    plt.plot(r[:, i, 0], r[:, i, 1], color = planet_sun[i][1], label = planet_sun[i][0])
plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')
    
radius = np.linalg.norm([np.mean(r[:1000, 0, 0]), np.mean(r[:1000, 0, 1])])
theta = np.linspace(0, 2*np.pi, N)
plt.plot(radius*np.cos(theta), radius*np.sin(theta), color = 'lightcoral', label = 'perfect circle')

plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.axis('equal')
plt.title("Flora's and our sun's orbit around their\ncenter of mass")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('Flora&sun_cm.pdf')


plt.plot(utils.AU_to_km(r[:, 1, 0]), utils.AU_to_km(r[:, 1, 1]), color = 'orange', label = 'Sun')
plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')

radius = utils.AU_to_km(np.linalg.norm([np.mean(r[:1000, 1, 0]), np.mean(r[:1000, 1, 1])]))
theta = np.linspace(0, 2*np.pi, N)
plt.plot(radius*np.cos(theta), radius*np.sin(theta), color = 'lightcoral', label = 'perfect circle')

plt.legend(loc = 'lower left')
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.axis('equal')
plt.title("Our sun's orbit around the center of mass")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('twobody_sunorbit_cm.pdf')

'''
the center of mass is in a focal point, so we see that both Flora and the sun has
slightly elliptical orbits with the center of mass in one of the focal points
'''

energy(int(3*1e4), dt, m, M, r, v)

angularmomentum(int(3*1e4), dt, m, M, r, v)





'''
C2. The Radial Velocity Curve
'''
'''
Task 1
'''

v_pec = np.array([- 1.5*10**(-3), 0.0])         # center of mass velocity relative to observer (peculiar velocity) [AU/yr]

N = 4*10**4
dt = 2*P/N

'''
we choose an inclination of 90 degrees, which means that our line of sight is 
parallell with the plane in which our orbit it situated
'''

v_sun = v[:, 1]                                 # the velocity of our sun relative to the center of mass [AU/yr]
t, v_real, v_obs = radial_velocity_curve(N, dt, v_sun, v_pec)

'''
Task 2
'''

velocity_data(t, v_obs, 'velocitydata.txt')

'''
we now want to calculate the mass of an extrasolar planet using the radial 
velocity method. we assume that the inclination is at 90 degrees, which means
sin(i) = 1. this gives us the smallest possible mass of the planet
'''

m_group = 1.3966746879342887e-06             # the actual mass of the planet the group used for their two-body system [M]
M_group = 4.4408144144136115                 # the actual mass of the group's sun [M]

est_m, vr_group = vr_from_group(m_group, M_group, 'velocitydata_fromgroup.txt')




'''
C3. The Light Curve
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
        t0 = i*dt*const.yr/(60*60)
        break

t, F_obs = light_curve(t0, N, m, M, F_rad, sun_rad, v_sun)
light_curve_data(t, F_obs, 'lightcurvedata.txt')

'''
Task 2 and 3
'''

m_group = m_group*m_sun                                # the actual mass of the group's planet [kg]         
M_group = M_group*m_sun                                # the actual mass of the group's star [kg]
radius = 5134045.72247952                              # the actual radius of the group's planet [m]
density = 3*m_group/(4*np.pi*radius**3)                # the actual density of the group's planet [kg/m^3] 
est_m = est_m*m_sun                                    # estimated smallest mass of the group's planet [kg]
vr_group = utils.AU_pr_yr_to_m_pr_s(vr_group)          # the estimated radial velocity of the group's star [m/s]

light_curve_from_group(M_group, radius, density, est_m, vr_group, 'lightcurvedata_fromgroup.txt')   





'''
C4. The Radial Velocity Curve with More Planets
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

D_r0 = np.array([init_pos[0, 0], init_pos[1, 0]])   # Doofenshmirtz' initial position [AU]
D_v0 = np.array([init_vel[0, 0], init_vel[1, 0]])   # Doofenshmirtz' initial velocity [AU/yr]

B_r0 = np.array([init_pos[0, 2], init_pos[1, 2]])   # Bubbles' initial position [AU]
B_v0 = np.array([init_vel[0, 2], init_vel[1, 2]])   # Bubbles' initial velocity [AU/yr]

A_r0 = np.array([init_pos[0, 6], init_pos[1, 6]])   # Aisha's initial position [AU]
A_v0 = np.array([init_vel[0, 6], init_vel[1, 6]])   # Aisha's initial velocity [AU/yr]

sun_r0 = np.array([0.0, 0.0])                       # our sun's initial position [AU]

m = [D_m, B_m, F_m, A_m]
r0 = np.array([D_r0, B_r0, F_r0, A_r0, sun_r0])
v0 = [D_v0, B_v0, F_v0, A_v0]

planets_momentum = np.zeros(2)
for i in range(4):
    planets_momentum += v0[i]*m[i]                  # the total linear momentum of the planet's we're studying [M*AU/yr]

sun_v0 = - planets_momentum/M                       # our sun's initial velocity [AU/yr]

cm_r = np.zeros(2)
cm_v = np.zeros(2)

m.append(M)
v0.append(sun_v0)

'''
changing frame of reference: cm_r is the positional vector of the center of
mass, which points from the origin and out. since our sun is located in the
origin of our current frame of reference, the positional vector points from
our sun to the center of mass
'''

for i in range(5):
    cm_r += m[i]/sum(m)*r0[i]             # center of mass position relative to the sun
    cm_v += m[i]/sum(m)*v0[i]             # center of mass velocity relative to the sun

r0 = r0 - cm_r                            # changing the initial positions so that they're relative to the center of mass
v0 = v0 - cm_v                            # changing the initial velocities so that they're relative to the center of mass

cm_r = np.zeros(2)                        # placing the center of mass in the origin
cm_v = np.zeros(2)

planet_m = np.sum(m[:4])                       # the sum of the planets' masses [M]               
planet_cm_r0 = np.zeros(2)
planet_cm_v0 = np.zeros(2)

for i in range(4):
    planet_cm_r0 += m[i]/planet_m*r0[i]        # the planets' center of mass position relative to the system's center of mass
    planet_cm_v0 += m[i]/planet_m*v0[i]        # the planets' center of mass velocity relative to the system's center of mass

sun_r0 = r0[-1]
sun_v0 = v0[-1]

planets_m = np.array(m[:4])
planets_r0 = r0[:4]
planets_v0 = np.array(v0[:4])

N = 20*10**4        # amount of time steps
dt = 20*P/N         # time step

r, v = nbody_orbits(N, dt, planets_m, M, planets_r0, planets_v0, planet_cm_r0, planet_cm_v0, sun_r0, sun_v0)

planets_sun = np.array([['Doofenshmirtz', 'black'], ['Bubbles', 'skyblue'], 
                        ['Flora', 'pink'], ['Aisha', 'darkorchid'], ['Sun', 'orange']])

for i in range(5):
    plt.plot(r[:, i, 0], r[:, i, 1], color = f'{planets_sun[i, 1]}', label = f'{planets_sun[i, 0]}')
    
plt.legend(loc = 'lower left')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.axis('equal')
plt.title("Our planets' and sun's orbit around their\ncommon center of mass")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('nbody&sun_cm.pdf')


plt.plot(utils.AU_to_km(r[:, -1, 0]), utils.AU_to_km(r[:, -1, 1]), color = 'orange', label = 'Sun')
plt.scatter(cm_r[0], cm_r[1], color = 'k', label = 'center of mass')

radius = utils.AU_to_km(np.linalg.norm([np.mean(r[:1000, -1, 0]), np.mean(r[:1000, -1, 1])]))
theta = np.linspace(0, 2*np.pi, N)
plt.plot(radius*np.cos(theta), radius*np.sin(theta), color = 'greenyellow', label = 'perfect circle')

plt.legend(loc = 'lower left')
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.axis('equal')
plt.title("Our sun's orbit around the center of mass")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('nbody_sunorbit_cm.pdf')


'''
Task 2
'''

N = 3*10**4             # number of time steps from the sun's radial velocity curve we want to analyze

v_pec = np.array([- 1.5*10**(-3), 0.0])         # center of mass velocity relative to observer (peculiar velocity) [AU/yr]
v_sun = v[:N, -1]                               # the velocity of our sun relative to the center of mass [AU/yr]

t, v_real, v_obs = radial_velocity_curve(N, dt, v_sun, v_pec, nbody=True)

velocity_data(t, v_obs, 'velocitydata_nbody.txt')

no = 4
mean_momentum_twobody = utils.m_pr_s_to_AU_pr_yr(vr_group)*M_group/m_sun  
est_no, vr_group = vr_from_group_nbody(no, M_group/m_sun, mean_momentum_twobody, 'velocitydata_nbody_fromgroup.txt')



'''
RESULTS:

        ENERGY AND ANGULAR MOMENTUM CONSERVATION:
    The relative error of the estimated energy of the system is 5.12e-07 %
    The relative error of the estimated angular momentum of the system is 2.84e-12 %

        RADIAL VELOCITY ANALYSIS:
    The estimated peculiar velocity of the group's system is -5e-06 AU/yr
    The estimated radial velocity of the group's star is 0.005 m/s
    The estimated revolution period for the group's star is 34.571 years
    The estimated time stamp of the sun's first peak in radial velocity is at t0 = 17.061 years
    The estimated mass of the planet is 1.401e-06 solar masses, while the actual mass is   solar masses.
    The relative error is 0.29 %.

        OUR LIGHT CURVE:
    Flora uses approximately 3.45 hours to fully eclipse the sun
    The relative light flux when Flora is fully eclipsing the sun is 0.9973

        LIGHT CURVE ANALYSIS:
    The estimated radius of the planet eclipsing the star is 0.00 km, while the actual radius is 5134.05 km. The relative error is 100.00 %
    The estimated density of the planet eclipsing the star is inf kg/m^3, while the actual density is 4899.45 kg/m^3. The relative error is inf %

        NBODY RADIAL VELOCITY ANALYSIS:
    The estimated peculiar velocity of the group's system is -5e-06 AU/yr
    The estimated radial velocity of the group's star is 0.213 m/s
    The estimated revolution period for the group's star is 165.429 years
    The estimated time stamp of the sun's first peak in radial velocity is at t0 = 47.694 years
    The estimated number of planets is 44.90, while the actual number of planets is 4.
'''