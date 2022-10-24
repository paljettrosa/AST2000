#EGEN KODE
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts #TODO
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [10978]) #TODO

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = 4*np.pi**2                                  # gravitation constant [AU**3yr**(-2)M**(-1)]
m_sun = const.m_sun                             # solar mass [kg]
AU = const.AU                                   # one astronomical unit [m]
day = const.day                                 # one day [s]
yr = const.yr                                   # one year [s]

M_p = system.masses                             # planets' masses [M]
M_s = system.star_mass                          # sun's mass [M]
r_s = np.array([0.0, 0.0])                      # sun's position [AU]
R_p = system.radii*1e3                          # planets' radiis [m]
T_p = system.rotational_periods*day             # planets' rotational periods [s]

A_box = L*L                                     # area of one gasbox [m**2]
A_spacecraft = mission.spacecraft_area          # area of our spacecraft's cross section [m**2]
N_box = int(A_spacecraft/A_box)                 # number of gasboxes   

spacecraft_m = mission.spacecraft_mass          # mass of rocket without fuel [kg]

#TODO nødvendig å inkludere dette eller importere r, N, dt?

@jit(nopython = True)
def simulate_orbits(N, dt, r0, v0, M):
    r = np.zeros((N, len(planets), 2))
    v = np.zeros((N, len(planets), 2))
    r[0] = r0
    v[0] = v0
    
    for i in range(N - 1):
        for j in range(len(planets)):
            
            g = - G*M/np.linalg.norm(r[i, j])**3*r[i, j]
            v[i+1, j] = v[i, j] + g*dt/2
            r[i+1, j] = r[i, j] + v[i+1, j]*dt
            
            g = - G*M/np.linalg.norm(r[i+1, j])**3*r[i+1, j]
            v[i+1, j] = v[i+1, j] + g*dt/2
            
    return r, v
        

@jit(nopython = True)
def fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, delta_v):
    
    '''
    we assume that the amount of fuel that the rocket loses during the speed boost
    is so minimal that we can define the rocket's acceleration as the total thrust force
    divided by it's total mass before the boost
    '''
    
    a = thrust_f/initial_m                                  # the rocket's acceleration [m/s**2]
    delta_t = delta_v/a                                     # time spent accelerating the rocket [s]
    tot_fuel_loss = abs(delta_t*fuel_loss_s*N_box)          # total fuel loss [kg]
    
    return tot_fuel_loss





'''
A. Simulating the Spacecraft's Trajectory
'''

def trajectory(t0, r0, v0, m, r_p, p_idx, N, dt, boosts):    
    startN = int(t0/dt)
    N = N - startN      
    
    for i in range(len(boosts[0])):
        boosts[0, i] = int(boosts[0, i]/dt)    
    
    r_p = r_p[:startN]
    r = np.zeros((N, 2))
    v = np.zeros((N, 2))
    
    for i in range(N):
        
        ''' checking if it's time for a speed boost '''
        
        for ti, dvx, dvy, dm in boosts:              
            if i == ti:                     
                v[i] = v[i] + [dvx, dvy]    # updating the rocket's velocity
                m = m - dm                  # updating the rocket's mass
        
        pos = r[i] - r_s
        fG = - G*m*M_s/np.linalg.norm(pos)**3*(pos)
        for j in range(len(planets)):
            pos = r[i] - r_p[i, j]
            fG -= G*m*M_p[j]/np.linalg.norm(pos)**3*(pos)
        g = fG/m
        
        v[i+1] = v[i] + g*dt/2
        r[i+1] = r[i] + v[i+1]*dt
        
        ''' checking if we've reached our destination '''
        
        distance = np.linalg.norm(r[i+1] - r_p[i+1, p_idx])                     # our spacecraft's distance from its destination
        l = np.linalg.norm(r[i+1])*np.sqrt(M_p[p_idx]/(10*M_s))                 # the maximal distance between the spacecraft and its destination in
                                                                                # order for it to be able to to perform an orbital injection manouver
        if distance <= l:
            #TODO krav oppfylt for at vi kan starte orbital injection?
            final_time = t0 + N*dt
            final_pos = r[i+1]
            final_vel = v[i+1]
            final_mass = m
            r = r[i+1:]
            v = v[i+1:]
            break
        
        #TODO sjekke om romskipet krasjer i en planet? eller sjekke om avstanden til 
        # en planet er mindre enn l for den planeten? printe at vi har gått i bane rundt fei planet isåfall?
        
        pos = r[i+1] - r_s
        fG = - G*m*M_s/np.linalg.norm(pos)**3*(pos)
        for j in range(len(planets)):
            pos = r[i+1] - r_p[i+1, j]
            fG -= G*m*M_p[j]/np.linalg.norm(pos)**3*(pos)
        g = fG/m

        v[i+1] = v[i+1] + g*dt/2
    
    return final_time, final_pos, final_vel, final_mass, distance #TODO, riktig distance som de spør om i 5D?





'''
B. Plan your Journey
'''

'''
we're adjusting our trajectory function from A so that it can take in arrays
containing time stamps when we want to boost our speed, and the difference in
both velocity components. we also have to check if our spacecraft has enough
fuel left after the boost is completed
'''

''' simulating orbits '''

a = system.semi_major_axes               # each planet's semi major axis [AU]

init_pos = system.initial_positions      # each planet's initial position [AU]
init_vel = system.initial_velocities     # each planet's initial velocity [AU/yr]

r0 = np.zeros((len(planets), 2))
v0 = np.zeros((len(planets), 2))
for i in range(len(planets)):
    r0[i] = np.array([init_pos[0][i], init_pos[1][i]])
    v0[i] = np.array([init_vel[0][i], init_vel[1][i]])

P = np.sqrt(4*np.pi**2*a[0]**3/(G*(M_s + M_p[0])))   # our home planet's revolution period [yr]

N = 40*10**4        # amount of time steps
dt_p = 40*P/N       # time step

r_p, v_p = simulate_orbits(N, dt_p, r0, v0, M_s)             


''' launching rocket at a chosen time, from a chosen position '''

phi = np.pi/2                           # launching from the north pole
t0 = 2                                  # launching two years after we started simulating the orbits
phi = 0
t0 = 0
#TODO fiks sånn at vi kan launche fra andre steder

dt = 1                                  # time step length for launch [s]
max_time = 20*60                        # maximum launch duration [s]


''' testing if we have enough fuel left to do our wanted speed boosts during trajectory '''

#TODO @jit(nopython = True)
def enough_fuel(N_H2, fuel_m, t0, phi, max_time, dt, r_p, dt_p, boosts=None): #TODO fjern dt, r_p og dt_p
    r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
    
    particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
    mean_f = f/steps                                        # the box force averaged over all time steps [N]
    fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]
                       
    thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]
    
    initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
    mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]
    
    ''' launching '''
    
    #r, v, rocketm_afterlaunch, bool_value = rocket_launch(t0, p_idx, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate, r_p, dt_p)

    r0 = utils.m_to_AU(np.array([np.cos(phi)*R_p[0], np.sin(phi)*R_p[0]])) + r_p[int(t0/dt_p)]
    mission.set_launch_parameters(thrust = thrust_f, 
                                  mass_loss_rate = mass_loss_rate, 
                                  initial_fuel_mass = fuel_m, 
                                  estimated_launch_duration = max_time, 
                                  launch_position = r0, 
                                  time_of_launch = t0)

    mission.launch_rocket()
    fuel_consumed = 0
    fuel_consumed, time_afterlaunch, pos_afterlaunch, vel_afterlaunch = shortcut.get_launch_results()
    mission.verify_launch_result(pos_afterlaunch)
    rocketm_afterlaunch = initial_m - fuel_consumed
    
    if fuel_consumed != 0:  
        if boosts != None:
            tot_fuel_loss = 0
            for i in range(len(boosts[0])):
                dv = np.linalg.norm(np.array([boosts[1, i], boosts[2, i]]))
                boosts[3, i] = fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, dv)
                tot_fuel_loss += boosts[3, i]
            
            rocketm_afterboosts = rocketm_afterlaunch - tot_fuel_loss
            if rocketm_afterboosts <= spacecraft_m:
                print("The rocket's mass is %.2f kg after completing the boosts! Bring more fuel with you" % rocketm_afterboosts)
            else:
                print("The rocket's mass is %.2f kg after completing the boosts! You're ready for travel :)" % rocketm_afterboosts)

                ''' converting to astronomical units '''
                
                rocketm_afterlaunch = rocketm_afterlaunch/m_sun
                boosts[1] = boosts[1]/AU/yr
                boosts[2] = boosts[2]/AU/yr
                boosts[3] = boosts[3]/m_sun
    
    return r, v, rocketm_afterlaunch, pos_afterlaunch, vel_afterlaunch, boosts

N_H2 = 5*10**6
fuel_m = 5*10**4

#TODO boosts = np.array([t1, t2, t3], [[dvx1, dvy1], [dvx2, dvy3], [dvx3, dvy3]], [0.0, 0.0, 0.0]) 
#TODO definer en slik array, der siste liste er liste over massen raketten mister fra hver speed boost
#TODO tidene er i år, dv'ene er i m/s, og massene blir først i kg etter enough_fuel
boosts = np.array([[2.3, 2.8, 3.4], [200, 150, 110], [120, 170, 230], [0.0, 0.0, 0.0]])

r, v, m_sc, pos_afterlaunch, vel_afterlaunch, boosts = enough_fuel(N_H2, fuel_m, t0, phi, max_time, dt, r_p[:, 0], dt_p, boosts)





'''
C. Sending the Spacecraft


r_p, v_p, dist_home_Buttercup, t0, home_pos, Buttercup_pos = simulate_orbits(N, dt_p, r0, v0, M_s, 3)
print(dist_home_Buttercup)
print(t0)
print(home_pos)
print(Buttercup_pos)

plt.plot(r_p[:, 0, 0], r_p[:, 0, 1], color = 'palevioletred')
plt.plot(r_p[:, 3, 0], r_p[:, 3, 1], color = 'olivedrab')
plt.plot(np.linspace(home_pos[0], Buttercup_pos[0], 100), np.linspace(home_pos[1], Buttercup_pos[1], 100), color = 'powderblue')
plt.axis('equal')
plt.show()

v_afterlaunch = np.array([1, 1])       #TODO finn denne
v_abs = np.linalg.norm(v_afterlaunch)

time = dist_home_Buttercup/v_abs
time_stamp = int((t0 + time)/dt_p)
new_Buttercup_pos = r_p[time_stamp, 3]


shortcuts = SpaceMissionShortcuts(mission, [10978])
mission.set_launch_parameters(thrust_f, fuel_loss_s, fuel_m, VELG EN LAUNCH DURATION FRA SIMULASJON, r[0], 2)
mission.launch_rocket()
fuel_consumed, time_after_launch, pos_after_launch, vel_after_launch = shortcuts.get_launch_results()

mission.verify_launch_result(pos_after_launch)

mission.take_picture("afterlaunch.png")

'''




'''
D. Orbit Stability
'''
'''
Task 1
'''
#final_time, final_pos, final_vel, final_mass, distance = trajectory(t0, r0, v0, m, r_p, p_idx, N, dt, boosts)
#TODO riktig distance? putt inn riktig variabler
'''
x, y = final_pos
vx, vy = final_vel
vr = (x*vx + y*vy)/np.sqrt(x**2 + y**2)
vr = (x*vx + y*vy)/distance
vtheta = (x*vy - y*vx)/(x**2 + y**2)
vtheta = (x*vy - y*vx)/distance**2
'''


'''
Task 2 and 3
'''
G = const.G                     # the gravitational constant in SI-units'

''' we use SI-units to avoid round-off errors '''

@jit(nopython = True)
def orbit_stability(t0, r0, v0, M, N, dt):       
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    
    r[0] = r0                                                           # initial position [m]
    v[0] = v0                                                           # initial velocity [m/s]
    
    a = np.zeros(3)
    b = np.zeros(3)
    e = np.zeros(3)
    apoapsis = np.zeros(3)
    periapsis = np.zeros(3)
    P = np.zeros(3)
    
    distance = np.linalg.norm(r[0])
    apoapsis[0] = distance                                            
    periapsis[0] = distance
    P[0] = 0
    
    orbits = 0
    tol = 1e-3
    for i in range(int(N) - 1):
        distance = np.linalg.norm(r[i])                                 # the current distance from the planet's center of mass [m]
        
        ''' updating our approximation of semi-major and semi-minor axes '''
        
        if distance >= apoapsis[orbits]:
            apoapsis[orbits] = distance
        
        if distance <= periapsis[orbits]:
            periapsis[orbits] = distance
        
        ''' checking if we've done a full orbit '''
        
        if i >= P[orbits]/dt*orbits + 200:
            if np.linalg.norm(r[i] - r0) <= tol:
                
                P[orbits] = i*dt
                a[orbits] = apoapsis[orbits] + periapsis[orbits]
                e[orbits] = apoapsis[orbits]/a[orbits] - 1
                b[orbits] = a[orbits]*np.sqrt(1 - e[orbits]**2)
                
                orbits = orbits + 1
                
                print(f'Approximations after {orbits} orbit(s):')
                print(f'    Semi-major axis:   {a[orbits]*1e-3:.2f} km')
                print(f'    Semi-minor axis:   {b[orbits]*1e-3:.2f} km')
                print(f'    Eccentricity:      {e[orbits]:.2f}')
                print(f'    Apoapsis:          {apoapsis[orbits]*1e-3:.2f} km')
                print(f'    Periapsis:         {periapsis[orbits]*1e-3:.2f} km')
                print(f'    Revolution period: {P[orbits]/yr:.2f} yrs')
                
                if orbits == 3:
                    r = r[i:]
                    v = v[i:]
                    break
                
                apoapsis[orbits] = distance                                            
                periapsis[orbits] = distance
                                        
        a = - G*M/np.linalg.norm(r[i])**3*r[i]                          # the rocket's acceleration at current time step [m/s**2]
        
        v[i+1] = v[i] + a*dt/2                                            
        r[i+1] = r[i] + v[i+1]*dt   

        a = - G*M/np.linalg.norm(r[i+1])**3*r[i+1]                      # the rocket's acceleration at current time step [m/s**2]

        v[i+1] = v[i+1] + a*dt/2 
        #TODO radiell og tangentiell hastighet? r = np.linalg og theta = tan-1(y/x)? bruk dette til å finne hastighet og akselerasjon?
    return r, v, a, b, e, apoapsis, periapsis, P #TODO nødvendig?


#TODO hvorfor bruker koden så lang tid?
#TODO fiks E i 1DEF sånn at F også funker
#TODO fiks ssview
#TODO fiks 4D orientation