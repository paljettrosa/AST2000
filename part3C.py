import numpy as np
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1F import max_time, dt, v_esc, initial_m, thrust_f, fuel_m, mass_loss_rate
from part2AB import dt_p, r_p, v_p

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

'''
C: Generalized Launch Codes
'''

AU = const.AU                               # astronomical unit of length [m]
M = const.m_sun                             # solar mass [kg]
yr = const.yr                               # one year [s]
AU_N = M*AU*yr**2                           # astronomical unit of force [N]

def gravity_g(r, rocket_m):                                                                 # generalized version of our gravity function from part 1
    theta = r[0]/np.linalg.norm(r)                                                          # angle between our current 
                                                                                            # positional vector and the x-axis
    abs_fG = - const.G*system.masses[0]*const.m_sun*rocket_m/np.linalg.norm(r)**2
    fG = np.array([abs_fG*np.cos(theta), abs_fG*np.sin(theta)])                             # vectorized gravitational pull
    return fG

def rocket_launch_g(t0, p_idx, dt_p, r_p, v_p, R_p, T_p, phi, v_esc, max_time, dt, thrust_f, initial_m, mass_loss_rate):      # generalized version of our rocket_launch function from part 1
    N = max_time/dt                                                     # number of time steps
    
    '''launching our rocket in the planet's frame of reference'''
    
    sim_launch_duration = 0                                             # duration of our simulated rocket launch [yr]
    rocket_m = initial_m                                                # the rocket's total mass [M]
    
    r0 = np.array([np.cos(phi)*R_p[p_idx], np.sin(phi)*R_p[p_idx]])
    v0 = np.array([0.0, 0.0]) 
    
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                                                           # initial position [m]
    v[0] = v0                                                           # initial velocity [m/s]
    
    for i in range(int(N) - 1):
        fG = gravity_g(r[i], rocket_m)                                  # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        
        v[i+1] = v[i] + a*dt/2                                            
        r[i+1] = r[i] + v[i+1]*dt   

        fG = gravity_g(r[i+1], rocket_m)                                # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]

        v[i+1] = v[i+1] + a*dt/2  
                       
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        if np.linalg.norm(v[i+1]) >= v_esc:                             # checking if the rocket has reached the escape velocity
            r = r[:i+1]
            v = v[:i+1]
            sim_launch_duration = (i*dt)                                # updating the duration of our simulated rocket launch
            break   
    
    '''changing to the sun's' frame of reference'''
    
    N_p = int(t0/dt_p)   
    
    v_orbitx = (r[-1, 0] - r[1, 0])/sim_launch_duration - v[-1, 0]      # the rocket's approximated horizontal velocity caused by the planet's orbital velocity [m/s]
    v_orbity = (r[-1, 1] - r[1, 1])/sim_launch_duration - v[-1, 1]      # the rocket's approximated vertical velocity caused by the planet's orbital velocity [m/s]
    
    omega = 2*np.pi/T_p[p_idx]                                          # the planet's rotational velocities [s**(-1)]
    v_rot = - R_p[p_idx]*omega                                          # our rocket's initial velocity caused by the planet's rotation [m/s]
    
    vx0 = v_orbitx + np.sin(phi)*v_rot
    vy0 = v_orbity + np.cos(phi)*v_rot
    v0 = np.array([vx0, vy0])                                           # our rocket's initial velocity relative to our sun [m/s]         
    
    r0_p = r_p[N_p, p_idx]                                              # the planet's initial positon relative to the sun [m]    
    r0 = r0 + r0_p                                                      # our rocket's launch position relative to our sun [m]
   
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                                                           # initial position [m]
    v[0] = v0                                                           # initial velocity [m/s]
    
    sim_launch_duration = 0                                             # duration of our simulated rocket launch [s]
    rocket_m = initial_m                                                # the rocket's total mass [kg]
    
    for i in range(int(N) - 1):
        fG = gravity_g(r[i] - r0_p, rocket_m)                           # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        
        v[i+1] = v[i] + a*dt/2                                            
        r[i+1] = r[i] + v[i+1]*dt   

        fG = gravity_g(r[i+1] - r0_p, rocket_m)                         # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]

        v[i+1] = v[i+1] + a*dt/2  
                       
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        if np.linalg.norm(v[i+1]) >= v_esc:                             # checking if the rocket has reached the escape velocity
            r = r[:i+1]
            v = v[:i+1]
            sim_launch_duration = (i*dt)                                # updating the duration of our simulated rocket launch
            break 
    
    print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][1]/10**3:g} km\nwhen it reaches the escape velocity")
    print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g}, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
    print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
    print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {rocket_m:g} kg, which means it lost a total of {initial_m - rocket_m:g} kg fuel\nduring the launch")
    
    ''' changing to astronomical units'''
    
    r0 = r0/AU
    r = r/AU
    v = v/AU/yr
    
    return r0, r, v

'''
phi tells us where on the planet we want to travel from, as it's the angle
between the rocket's launch position and the planet's equatorial
'''
'''
testing our generalized function with the launch parameters from part 1
'''

phi = 0
p_idx = 0
t0 = 0
R_p = system.radii*1e3
T_p = system.rotational_periods*const.day 

dt_p = dt_p*yr
r_p = utils.AU_to_m(r_p)
v_p = utils.AU_pr_yr_to_m_pr_s(v_p)

r0, r, v, = rocket_launch_g(t0, p_idx, dt_p, r_p, v_p, R_p, T_p, phi, v_esc, max_time, dt, thrust_f, initial_m, mass_loss_rate)

mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = 1000, 
                              launch_position = r0, 
                              time_of_launch = t0)

mission.launch_rocket()

'''
trying with other parameters
'''

phi = np.pi/2                           # launching from the north pole of our planet
p_idx = 3                               # launching from Buttercup
t0 = 2*yr                               # launching two years after we started simulating the orbits

r0, r, v, = rocket_launch_g(t0, p_idx, dt_p, r_p, v_p, R_p, T_p, phi, v_esc, max_time, dt, thrust_f, initial_m, mass_loss_rate)

'''
FRA EKVATOR PÅ HJEMPLANETEN

The rocket's position is at x = 5.2972e+08 km, y = -1108.84 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 9063.2, it's
velocity has a horisontal component of 7614.01 m/s and a vertical
component of -4916.14 m/s
The simulated rocket launch took 603 seconds, which is
approximately 10 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 8060.52 kg, which means it lost a total of 23039.5 kg fuel
during the launch
Rocket was moved up by 4.50388e-06 m to stand on planet surface.
New launch parameters set.
Launch completed, reached escape velocity in 553.75 s


FRA NORDPOLEN PÅ BUTTERCUP

The rocket's position is at x = -1.57771e+09 km, y = 6.75748e+08 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 9022.05, it's
velocity has a horisontal component of 8972.61 m/s and a vertical
component of 943.205 m/s
The simulated rocket launch took 649 seconds, which is
approximately 10 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 6305.86 kg, which means it lost a total of 24794.1 kg fuel
during the launch
'''