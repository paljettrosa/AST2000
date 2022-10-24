#EGEN KODE
import numpy as np
from tqdm import trange
from numba import jit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = const.G                                                 # gravitation constant [m**3s**(-2)kg**(-1)]
M = system.masses[0]*const.m_sun                            # mass of our home planet [kg]
R = system.radii[0]*10**3                                   # our planet's radius [m]
x0 = utils.AU_to_m(system.initial_positions[0][0])          # our planet's initial x-position relative to our sun [m]
y0 = utils.AU_to_m(system.initial_positions[1][0])          # our planet's initial y-position relative to our sun [m]
distance_ps = np.linalg.norm(np.array([x0, y0]))            # the distance from our planet to our sun [m]                                          

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


def rocket_launch(r0, v0, max_time, dt, thrust_f, initial_m, mass_loss_rate):
    sim_launch_duration = 0             # duration of our simulated rocket launch [s]
    rocket_m = initial_m                # the rocket's total mass [kg]
    N = max_time/dt                     # number of time steps
    
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                           # initial position [m]
    v[0] = v0                           # initial velocity [m/s]
    
    for i in trange(int(N) - 1):
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
        v[i+1] = v[i] + a*dt                                            # updated velocity
        r[i+1] = r[i] + v[i+1]*dt                                       # updated position
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        if rocket_m <= 0:                                               # checking if we run out of fuel
            print('Ran out of fuel!')
            break
        
        if np.linalg.norm(v[i+1]) >= v_esc:           # checking if the rocket has reached the escape velocity
            r = r[:i+1]
            v = v[:i+1]
            sim_launch_duration = i*dt                # updating the duration of our simulated rocket launch
            
            print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][1]/10**3:g} km\nwhen it reaches the escape velocity")
            print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g} m/s, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
            print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
            print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {rocket_m:g} kg, which means it lost a total of {initial_m - rocket_m:g} kg fuel\nduring the launch\n")
            break                 
        
    return r, v, sim_launch_duration




'''
D. The Rocket Engine's Performance
'''
N_H2 = 2*10**6                          # number of H_2 molecules

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)

particles_s = exiting/time              # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                        # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2          # the total fuel loss per second [kg/s]

A_box = L*L                             # area of one gasbox [m**2]
A_spacecraft = mission.spacecraft_area  # area of our spacecraft's cross section [m**2]
N_box = int(A_spacecraft/A_box)         # number of gasboxes                   
thrust_f = N_box*mean_f                 # the combustion chamber's total thrust force [N]

print(f'There are {particles_s*N_box:g} particles exiting the combustion chamber per second')
print(f'The combustion chamber exerts a thrust of {thrust_f:g} N')
print(f'The combustion chamber loses a mass of {fuel_loss_s*N_box:g} kg/s\n')

spacecraft_m = mission.spacecraft_mass  # mass of rocket without fuel [kg]
fuel_m = 1.5*10**4                      # mass of feul [kg]
initial_m = spacecraft_m + fuel_m       # initial rocket mass [kg]

delta_v = 10**4                         # change in the rocket's velocity [m/s]

tot_fuel_loss = fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, delta_v)

print(f'The rocket uses a total of {tot_fuel_loss:g} kg fuel to boost its speed {delta_v:g} m/s')




'''
E. Simulating a Rocket Launch
'''
'''
let's assume that we wish to launch our rocket from the equator, on the side of the
planet facing away from our sun. then our initial position will be as follows
'''
r0 = np.array([R, 0.0])

'''
when our rocket is moving away from our home planet, it's still within it's gravitational
field, and therefore moves around our sun with the same velocity as our planet does. because
of this, we ignore this contribution to our rocket's initial velocity. the contribution
coming from our planet's rotational velocity is still important to include, as our rocket
stops rotating almost immediately after leaving our planet's surface. our rocket will then
have a vertical velocity component relative to our planet, assuming that our planet rotates
around the x-axis when it's in it's initial position
'''

T = system.rotational_periods[0]*24*60*60               # our planet's rotational period [s]
omega = 2*np.pi/T                                       # our planet's rotational velocity [s**(-1)]
v_rot = - R*omega                                       # our rocket's initial velocity caused by our planet's rotation
                                                        # assuming that our planet's rotational velocity is positive [m/s]
v0 = np.array([0.0, v_rot])

dt = 1                                                  # time step [s]
max_time = 20*60                                        # maximum launch time [s]

initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

r, v, sim_launch_duration = rocket_launch(r0, v0, max_time, dt, thrust_f, initial_m, mass_loss_rate)




'''
F. Entering the Solar System
'''

''' changing reference system '''

r0 = np.array([x0 + R, y0])                             # our rocket's initial position relative to our sun [m]

end_y = r[-1][1]
v_orbit = end_y/sim_launch_duration                     # our rocket's initial velocity caused by our planet's orbital
                                                        # velocity, approximated by our simulation [m/s]
v0 = np.array([0.0, v_orbit + v_rot])                   # our rocket's initial velocity relative to our sun [m/s]

''' regulating launch parameters for the actual launch '''

N_H2 = 6*10**6  
r_particles, v_particles, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
particles_s = exiting/time              
mean_f = f/steps                        
fuel_loss_s = particles_s*m_H2
mass_loss_rate = N_box*fuel_loss_s                                
thrust_f = N_box*mean_f                
fuel_m = 3*fuel_m   
initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box                                             

''' launching '''

r, v, sim_launch_duration = rocket_launch(r0, v0, max_time, dt, thrust_f, initial_m, mass_loss_rate)

mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = 1000, 
                              launch_position = utils.m_to_AU(r0), 
                              time_of_launch = 0.0)

mission.launch_rocket()

'''
position_after_launch = np.array([utils.m_to_AU(r[-1][0]), utils.m_to_AU(r[-1][1])])
mission.verify_launch_result(position_after_launch)
'''

from ast2000tools.shortcuts import SpaceMissionShortcuts
shortcut = SpaceMissionShortcuts(mission, [10978])

fuel_consumed, time_after_launch, pos_after_launch, vel_after_launch = shortcut.get_launch_results()
mission.verify_launch_result(pos_after_launch)


'''
FROM D:
    
There are 1.1314e+28 particles exiting the combustion chamber per second
The combustion chamber exerts a thrust of 156494 N
The combustion chamber loses a mass of 37.8731 kg/s
The rocket uses a total of 38963.5 kg fuel to boost its speed 10000 m/s


FROM E:
    
The rocket uses a total of 38963.5 kg fuel to boost its speed 10000 m/s
The rocket's position is at x = 7257.55 km, y = -750.01 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 9988.5 m/s, it's
velocity has a horisontal component of 9438.36 m/s and a vertical
component of -3269.17 m/s
The simulated rocket launch took 398 seconds, which is
approximately 6 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 988.622 kg, which means it lost a total of 15111.4 kg fuel
during the launch


FROM F:
    
    SIMULATION RESULTS:

The rocket's position is at x = 5.29721e+08 km, y = -1373.6 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 10073.5 m/s, it's
velocity has a horisontal component of 8767.07 m/s and a vertical
component of -4961.19 m/s
The simulated rocket launch took 374 seconds, which is
approximately 6 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 3438.3 kg, which means it lost a total of 42661.7 kg fuel
during the launch

    LAUNCH RESULTS:
    
Rocket was moved up by 4.50388e-06 m to stand on planet surface.
New launch parameters set.
Launch completed, reached escape velocity in 390.68 s.
Your spacecraft position deviates too much from the correct position.

    VERIFICATION RESULTS:
        
The deviation is approximately 7.9315e-05 AU.
Make sure you have included the rotation and orbital velocity of your home planet.
Note that units are AU and relative the the reference system of the star.
'''