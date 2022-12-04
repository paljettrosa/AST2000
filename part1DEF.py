#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)
plt.rcParams.update({'font.size': 12})

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = const.G                                                 # gravitation constant [m**3s**(-2)kg**(-1)]
M = system.masses[0]*const.m_sun                            # mass of our home planet [kg]
R = system.radii[0]*1e3                                     # our planet's radius [m]

x0 = utils.AU_to_m(system.initial_positions[0][0])          # our planet's initial x-position relative to our sun [m]
y0 = utils.AU_to_m(system.initial_positions[1][0])          # our planet's initial y-position relative to our sun [m]
distance_ps = np.linalg.norm(np.array([x0, y0]))            # the distance from our planet to our sun [m]

spacecraft_m = spacecraft_m = mission.spacecraft_mass       # mass of rocket without fuel [kg]
spacecraft_A = mission.spacecraft_area                      # area of our spacecraft's cross section [m**2]                                          


@jit(nopython = True)
def fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, delta_v): 

    '''
    function for calculating the amount of fuel
    our rocket engine uses to boost an arbitrary Δv
    '''

    a = thrust_f/initial_m                                  # the rocket's acceleration [m/s**2]
    delta_t = delta_v/a                                     # time spent accelerating the rocket [s]
    tot_fuel_loss = abs(delta_t*fuel_loss_s*N_box)          # total fuel loss [kg]
    
    return tot_fuel_loss


def rocket_launch(r0, v0, max_time, dt, thrust_f, initial_m, mass_loss_rate):

    '''
    simulating a rocket launch
    '''

    sim_launch_duration = 0             # duration of our simulated rocket launch [s]
    rocket_m = initial_m                # the rocket's total mass [kg]
    N = max_time/dt                     # number of time steps
    
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                           # initial position [m]
    v[0] = v0                           # initial velocity [m/s]
    
    for i in range(int(N) - 1):
        distance_rp = np.linalg.norm(r[i])                              # the current distance from our point of reference [m]
    
        ''' checking what referance system we're using '''
        
        if distance_rp - distance_ps < 0:                               # the planet's reference system
            v_esc = np.sqrt(2*G*M/distance_rp)                          # the current escape velocity [m/s]
            fG = - G*M*rocket_m/np.linalg.norm(r[i])**3*r[i]            # the gravitational pull from our home planet [N]
            
        elif distance_rp - distance_ps >= 0:                            # the sun's reference system
            v_esc = np.sqrt(2*G*M/(distance_rp - distance_ps))          # the current escape velocity [m/s]
            pos = r[i] - np.array([x0, y0])
            fG = - G*M*rocket_m/np.linalg.norm(pos)**3*pos              # the gravitational pull from our home planet [N]
            
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        v[i+1] = v[i] + a*dt/2                                          # updated velocity
        r[i+1] = r[i] + v[i+1]*dt                                       # updated position

        if distance_rp - distance_ps < 0:                               # the planet's reference system
            v_esc = np.sqrt(2*G*M/distance_rp)                          # the current escape velocity [m/s]
            fG = - G*M*rocket_m/np.linalg.norm(r[i+1])**3*r[i+1]        # the gravitational pull from our home planet [N]
            
        elif distance_rp - distance_ps >= 0:                            # the sun's reference system
            v_esc = np.sqrt(2*G*M/(distance_rp - distance_ps))          # the current escape velocity [m/s]
            pos = r[i+1] - np.array([x0, y0])
            fG = - G*M*rocket_m/np.linalg.norm(pos)**3*pos              # the gravitational pull from our home planet [N]
        
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        v[i+1] = v[i+1] + a*dt/2                                        # updated velocity   
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        
        if thrust_f < np.linalg.norm(fG):                               # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        if rocket_m <= spacecraft_m:                                    # checking if we run out of fuel
            print('Ran out of fuel!')
            break
        
        if np.linalg.norm(v[i+1] - v[0]) >= v_esc:    # checking if the rocket has reached the escape velocity
            r = r[:i+2]
            v = v[:i+2]
            sim_launch_duration = (i+1)*dt            # updating the duration of our simulated rocket launch
            
            print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][1]/10**3:g} km\nwhen it reaches the escape velocity")
            print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g} m/s, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
            print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
            print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {rocket_m:g} kg, which means it lost a total of {initial_m - rocket_m:g} kg fuel\nduring the launch")
            break                 
        
    return r, v, sim_launch_duration, r[-1]




'''
D. The Rocket Engine's Performance
'''

N_H2 = 5*10**6                          # number of H_2 molecules

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)

particles_s = exiting/time              # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                        # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2          # the total fuel loss per second [kg/s]

box_A = L*L                             # area of one gasbox [m**2]
N_box = int(spacecraft_A/box_A)         # number of gasboxes                   
thrust_f = N_box*mean_f                 # the combustion chamber's total thrust force [N]

print(f'There are {particles_s*N_box:g} particles exiting the combustion chamber per second')
print(f'The combustion chamber exerts a thrust of {thrust_f:g} N')
print(f'The combustion chamber loses a mass of {fuel_loss_s*N_box:g} kg/s\n')

spacecraft_m = mission.spacecraft_mass  # mass of rocket without fuel [kg]
fuel_m = 4*10**4                        # mass of feul [kg]
initial_m = spacecraft_m + fuel_m       # initial rocket mass [kg]

delta_v = 10**3                         # change in the rocket's velocity [m/s]

tot_fuel_loss = fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, delta_v)

print(f'The rocket uses a total of {tot_fuel_loss:g} kg fuel to boost its speed {delta_v:g} m/s')




'''
E. Simulating a Rocket Launch
'''

r0 = np.array([R, 0.0])

T = system.rotational_periods[0]*24*60*60               # our planet's rotational period [s]
omega = 2*np.pi/T                                       # our planet's rotational velocity [s**(-1)]
v_rot = R*omega                                         # our rocket's initial velocity relative tho the planet  
                                                        # due to its rotational velocity [m/s]
v0 = np.array([0.0, v_rot])

dt = 1                                                  # time step [s]
max_time = 20*60                                        # maximum launch time [s]

initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

r, v, sim_launch_duration, final_pos = rocket_launch(r0, v0, max_time, dt, thrust_f, initial_m, mass_loss_rate)

distance = np.linalg.norm(final_pos) - R
print(f"The spacecraft's distance from the surface of Doofenshmirtz is {distance*1e-3:.2f} km when reaching escape velocity\n")

plt.plot((r[:, 0] - R)*1e-3, r[:, 1]*1e-3, color = 'mediumvioletred', label = 'trajectory')
plt.plot([0, 0], ([3*np.max(np.abs(r[:, 1]))/2*1e-3, - np.max(np.abs(r[:, 1]))/2*1e-3]), color = 'mediumorchid', label = 'surface')
plt.legend()
plt.ylabel('y-position [km]')
plt.xlabel('x-position [km]')
plt.title("The rocket's trajectory throughout the\nsimulated launch sequence")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('trajectory_launch_planetref.pdf')

t = np.linspace(0, sim_launch_duration, len(r))
plt.plot(t, np.linalg.norm(v, axis=1)*1e-3, color = 'cornflowerblue')
plt.ylabel(r'absolute velocity ($v$) [$\frac{km}{s}$]')
plt.xlabel('time [s]')
plt.title("The rocket's absolute velocity throughout\nthe simulated launch sequence")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('velocity_launch_planetref.pdf')




'''
F. Entering the Solar System
'''

''' changing reference system '''

r0 = np.array([x0 + R, y0])                             # our rocket's initial position relative to our sun [m]

v_orbit = utils.AU_pr_yr_to_m_pr_s([system.initial_velocities[0, 0], system.initial_velocities[1, 0]])
v0 = np.array([v_orbit[0], v_orbit[1] + v_rot])         # our rocket's initial velocity relative to our sun [m/s]

''' regulating launch parameters for the next launch '''
  
N_H2 = 6*10**6
r_particles, v_particles, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
particles_s = exiting/time              
mean_f = f/steps                        
fuel_loss_s = particles_s*m_H2
mass_loss_rate = N_box*fuel_loss_s                                
thrust_f = N_box*mean_f    
fuel_m = 4.5*10**4  
initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box                                             

''' launching '''

r, v, sim_launch_duration, final_pos = rocket_launch(r0, v0, max_time, dt, thrust_f, initial_m, mass_loss_rate)

x1 = x0 + utils.AU_pr_yr_to_m_pr_s(x0)*sim_launch_duration                 # Doofenshmirtz' updated x-coordinate
y1 = y0 + utils.AU_pr_yr_to_m_pr_s(y0)*sim_launch_duration                 # Doofenshmirtz' updated y-coordinate
distance = np.linalg.norm(final_pos - np.array([x1, y1])) - R
print(f"The spacecraft's distance from the surface of Doofenshmirtz is {distance*1e-3:.2f} km when reaching escape velocity")
print(f"The spacecraft's updated coordinates relative to Doofenshmirtz' surface are then [{(r[-1, 0] - x1 - R)*1e-3:.2f}, {(r[-1, 1] - y1)*1e-3:.2f}]\n")

plt.plot(r[:, 0]*1e-3, r[:, 1]*1e-3, color = 'mediumvioletred', label = 'trajectory')
plt.plot([r0[0]*1e-3, r0[0]*1e-3], ([3*np.max(np.abs(r[:, 1]))/2, - np.max(np.abs(r[:, 1]))/2]-r0[1])*1e-3, color = 'mediumorchid', label = 'surface')
plt.legend()
plt.ylabel('y-position [km]')
plt.xlabel('x-position [km]')
plt.title("The rocket's trajectory throughout the\nsimulated launch sequence")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('trajectory_launch_sunref.pdf')

t = np.linspace(0, sim_launch_duration, len(r))
plt.plot(t, np.linalg.norm(v, axis=1)*1e-3, color = 'cornflowerblue')
plt.ylabel(r'absolute velocity ($v$) [$\frac{km}{s}$]')
plt.xlabel('time [s]')
plt.title("The rocket's absolute velocity throughout\nthe simulated launch sequence")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('velocity_launch_sunref.pdf')


mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = 1000, 
                              launch_position = utils.m_to_AU(r0), 
                              time_of_launch = 0.0)

mission.launch_rocket()

'''
mission.verify_launch_result(utils.m_to_AU(final_pos))
'''

from ast2000tools.shortcuts import SpaceMissionShortcuts
shortcut = SpaceMissionShortcuts(mission, [10978])

fuel_consumed, time_after_launch, pos_after_launch, vel_after_launch = shortcut.get_launch_results()
mission.verify_launch_result(pos_after_launch)

plt.plot(r[:, 0]*1e-3, r[:, 1]*1e-3, color = 'mediumvioletred', label = 'trajectory')
plt.plot([r0[0]*1e-3, r0[0]*1e-3], ([utils.AU_to_km(pos_after_launch[1]), - (np.max(np.abs(r[:, 1]))/2 - r0[1])*1e-3]), color = 'mediumorchid', label = 'surface')
plt.scatter(utils.AU_to_km(pos_after_launch[0]), utils.AU_to_km(pos_after_launch[1]), marker = 'x', color = 'deeppink', label = 'expected position')
plt.legend()
plt.ylabel('y-position [km]')
plt.xlabel('x-position [km]')
plt.title("The rocket's trajectory throughout the simulated\nlaunch sequence, with the expected position")
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig('launch_correctpos.pdf')





'''
RESULTS:

    FROM D: 
There are 2.8297e+28 particles exiting the combustion chamber per second
The combustion chamber exerts a thrust of 391078 N
The combustion chamber loses a mass of 94.7226 kg/s
The rocket uses a total of 9954.4 kg fuel to boost its speed 1000 m/s


    FROM E: 
The rocket's position is at x = 7035.95 km, y = 150.36 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 10475 m/s, it's
velocity has a horisontal component of 10469.8 m/s and a vertical
component of 329.344 m/s
The simulated rocket launch took 421 seconds, which is
approximately 7 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 1267.07 kg, which means it lost a total of 39832.9 kg fuel
during the launch
The spacecraft's distance from the surface of Doofenshmirtz is 731.40 km 
when reaching escape velocity


    FROM F:
    
    SIMULATION RESULTS:
The rocket's position is at x = 5.29721e+08 km, y = 10125.9 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 27764.2 m/s, it's
velocity has a horisontal component of 10161.7 m/s and a vertical
component of 25837.8 m/s
The simulated rocket launch took 384 seconds, which is
approximately 6 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 2453.99 kg, which means it lost a total of 43646 kg fuel
during the launch
The spacecraft's distance from the surface of Doofenshmirtz is 6093.78 km 
when reaching escape velocity

    LAUNCH RESULTS:
Rocket was moved up by 4.50388e-06 m to stand on planet surface.
New launch parameters set.
Launch completed, reached escape velocity in 391.71 s.
Your spacecraft position deviates too much from the correct position.

    VERIFICATION RESULTS:
The deviation is approximately 2.67408e-06 AU.
Make sure you have included the rotation and orbital velocity of your home planet.
Note that units are AU and relative the the reference system of the star.

    WITH SHORTCUT:
Your spacecraft position was satisfyingly calculated. Well done!
*** Achievement unlocked: No free launch! ***
'''





