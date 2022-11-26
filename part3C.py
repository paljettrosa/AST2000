#EGEN KODE
import numpy as np
import matplotlib.pyplot as plt
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

G = const.G                                         # gravitation constant [m**3s**(-2)kg**(-1)]
AU = const.AU                                       # one astronomical unit [m]
yr = const.yr                                       # one year [s]

m_p = system.masses*const.m_sun                     # planets' masses [kg]
R_p = system.radii*1e3                              # planets' radiis [m]
T_p = system.rotational_periods*const.day           # planets' rotational periods [s]

spacecraft_m = spacecraft_m = mission.spacecraft_mass                  # mass of rocket without fuel [kg]
spacecraft_A = mission.spacecraft_area                                 # area of our spacecraft's cross section [m**2]

f = np.load(r'/Users/paljettrosa/Documents/AST2000/planet_trajectories.npz')
times = f['times']
dt_p = times[1]*yr
r_p = np.einsum("ijk->kji", f['planet_positions'])*AU


'''
C: Generalized Launch Codes
'''

def rocket_launch_generalized(t0, p_idx, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate):     # generalized version of our rocket_launch function from part 1

    ''' we use SI-units during the calculations to avoid as many round-off errors as possible '''    

    N = max_time/dt                                                     # number of time steps
    sim_launch_duration = 0                                             # duration of our simulated rocket launch [s]
    planet_m = m_p[p_idx]                                               # the planet's mass [kg]
    rocket_m = initial_m                                                # the rocket's total mass [kg]
    
    ''' launching our rocket in the planet's frame of reference '''
    
    r0 = np.array([np.cos(phi)*R_p[p_idx], np.sin(phi)*R_p[p_idx]])
    
    omega = 2*np.pi/T_p[p_idx]                                          # the planet's rotational velocity [s**(-1)]
    v_rot = - R_p[p_idx]*omega                                          # our rocket's initial velocity caused by the planet's rotation [m/s]
    v0 = np.array([- np.sin(phi)*v_rot, np.cos(phi)*v_rot]) 
    
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                                                           # initial position [m]
    v[0] = v0                                                           # initial velocity [m/s]
    
    for i in range(int(N) - 1):
        distance_rp = np.linalg.norm(r[i])                              # the current distance from the planet's center of mass [m]
        v_esc = np.sqrt(2*G*planet_m/distance_rp)                       # the current escape velocity [m/s]
        
        fG = - G*planet_m*rocket_m/np.linalg.norm(r[i])**3*r[i]         # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        
        v[i+1] = v[i] + a*dt/2                                            
        r[i+1] = r[i] + v[i+1]*dt   

        fG = - G*planet_m*rocket_m/np.linalg.norm(r[i+1])**3*r[i+1]     # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]

        v[i+1] = v[i+1] + a*dt/2  
                       
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        if rocket_m <= spacecraft_m:                                    # checking if we run out of fuel
            print('Ran out of fuel!')
            break
        
        if np.linalg.norm(v[i+1]) >= v_esc:                             # checking if the rocket has reached the escape velocity
            r = r[:i+1]
            v = v[:i+1]
            sim_launch_duration = (i*dt)                                # updating the duration of our simulated rocket launch
            break   
    
    '''changing to the sun's' frame of reference'''                   
    
    N_p = int(t0/dt_p)                                                  # finding out at which time step we start the rocket launch from
    x_p = r_p[N_p, p_idx, 0]
    y_p = r_p[N_p, p_idx, 1]
    distance_ps = np.linalg.norm(np.array([x_p, y_p]))                  # the distance from the planet to our sun at the current time [m]                            
    
    v_orbitx = (r[-1, 0] - r[1, 0])/sim_launch_duration                 # the rocket's approximated horizontal velocity caused by the planet's orbital velocity [m/s]
    v_orbity = (r[-1, 1] - r[1, 1])/sim_launch_duration                 # the rocket's approximated vertical velocity caused by the planet's orbital velocity [m/s]
    
    vx0 = v_orbitx - np.sin(phi)*v_rot
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
        distance_rp = np.linalg.norm(r[i])                              # the current distance from our point of reference [m]
        v_esc = np.sqrt(2*G*planet_m/(distance_rp - distance_ps))       # the current escape velocity [m/s]
        
        pos = r[i] - r0_p
        fG = - G*planet_m*rocket_m/np.linalg.norm(pos)**3*pos           # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        
        v[i+1] = v[i] + a*dt/2                                            
        r[i+1] = r[i] + v[i+1]*dt   
        
        pos = r[i+1] - r0_p
        fG = - G*planet_m*rocket_m/np.linalg.norm(pos)**3*pos           # the gravitational pull from the chosen planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]

        v[i+1] = v[i+1] + a*dt/2  
                       
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        if rocket_m <= spacecraft_m:                                               # checking if we run out of fuel
            print('Ran out of fuel!')
            break
        
        if np.linalg.norm(v[i+1]) >= v_esc:                             # checking if the rocket has reached the escape velocity
            r = r[:i+2]
            v = v[:i+2]
            sim_launch_duration = (i+1)*dt                              # updating the duration of our simulated rocket launch
            
            print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][1]/10**3:g} km\nwhen it reaches the escape velocity")
            print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g}, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
            print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
            print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {rocket_m:g} kg, which means it lost a total of {initial_m - rocket_m:g} kg fuel\nduring the launch")
            break 
    
    ''' changing to astronomical units'''
    
    r0 = utils.m_to_AU(r0)
    r = utils.m_to_AU(r)
    v = utils.m_pr_s_to_AU_pr_yr(v)
    
    return r0, r, v, sim_launch_duration




''' testing our generalized function with the launch parameters from part 1 '''

N_H2 = 6*10**6                                          # number of H_2 molecules

r_particles, v_particles, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)

particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                                        # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]

box_A = L*L                                             # area of one gasbox [m**2]
N_box = int(spacecraft_A/box_A)                         # number of gasboxes                   
thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s] 

fuel_m = 4.5*10**4                                      # mass of fuel [kg]
initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]

dt = 1                                  # time step [s]
max_time = 20*60                        # maximum launch time [s]

'''
phi tells us where on the planet we want to travel from, as it's the angle
between the rocket's launch position and the planet's equatorial
'''

phi = 0                                 # launching from the equatorial on the side of the planet facing away from the sun
p_idx = 0                               # launching from the home planet
t0 = 0                                  # launching at the beginning of the simulation of planetary orbits

r0, r, v, sim_launch_duration = rocket_launch_generalized(t0, p_idx, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate)

mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = 1000, 
                              launch_position = r0, 
                              time_of_launch = t0)

mission.launch_rocket()





''' 'trying with other parameters '''

N_H2 = 5*10**6                                          # number of H_2 molecules

r_particles, v_particles, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)

particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                                        # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]
             
thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s] 

fuel_m = 4*10**4                                        # mass of feul [kg]
initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]

phi = np.pi/2                           # launching from ϕ = π/2
p_idx = 3                               # launching from Buttercup
t0 = 2                                  # launching two years after we started simulating the orbits

r0, r, v, sim_launch_duration = rocket_launch_generalized(t0, p_idx, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate)




'''
RESULTS

    LAUNCHING FROM ϕ = 0 ON OUR HOME PLANET AT YEAR 0:

        SIMULATION RESULTS:
    The rocket's position is at x = 5.29722e+08 km, y = -269.009 km
    when it reaches the escape velocity
    When the rocket reaches it's escape velocity of 9890.75, it's
    velocity has a horisontal component of 9867.76 m/s and a vertical
    component of -673.895 m/s
    The simulated rocket launch took 379 seconds, which is
    approximately 6 minutes
    When the rocket reached it's escape velocity, it's total mass was
    down to 3067.61 kg, which means it lost a total of 43032.4 kg fuel
    during the launch
        ACTUAL LAUNCH RESULTS: 
    Rocket was moved up by 4.50388e-06 m to stand on planet surface.
    New launch parameters set.
    Launch completed, reached escape velocity in 391.43 s.


    LAUNCHING FROM ϕ = π/2 ON BUTTERCUP TWO YEARS AFTER THE
    PLANETARY ORBIT SIMULATION STARTED:

        SIMULATION RESULTS:
    The rocket's position is at x = -1.0059e+09 km, y = 1.40125e+09 km
    when it reaches the escape velocity
    When the rocket reaches it's escape velocity of 15654.8, it's
    velocity has a horisontal component of 14814.8 m/s and a vertical
    component of -5058.93 m/s
    The simulated rocket launch took 413 seconds, which is
    approximately 6 minutes
    When the rocket reached it's escape velocity, it's total mass was
    down to 2055.8 kg, which means it lost a total of 39044.2 kg fuel
    during the launch
'''