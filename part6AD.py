#EGEN KODE
#KANDIDATER 15361 & 15384
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts 
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps
from part4BCD import find_phi0, sc_velocity
from part5AB import fuel_consumption, r_Buttercup, time_stamp, N_H2, fuel_m, t0, phi, home_pos, max_time, times, exact_planet_positions

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)
shortcut = SpaceMissionShortcuts(mission, [10978]) 

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G_SI = const.G                                  # gravitation constant in SI-units [m**3s**(-2)kg**(-1)]
m_sun = const.m_sun                             # solar mass [kg]
AU = const.AU                                   # one astronomical unit [m]
day = const.day                                 # one day [s]
yr = const.yr                                   # one year [s]

M_p = system.masses                             # planets' masses [M]
M_p_SI = M_p*m_sun                              # planets' masses in SI-units [kg]
R_p = system.radii*1e3                          # planets' radiis [m]
r_s = np.array([0.0, 0.0])                      # sun's position [AU]
T_p = system.rotational_periods*day             # planets' rotational periods [s]

A_box = L*L                                     # area of one gasbox [m**2]
A_spacecraft = mission.spacecraft_area          # area of our spacecraft's cross section [m**2]
N_box = int(A_spacecraft/A_box)                 # number of gasboxes   

spacecraft_m = mission.spacecraft_mass          # mass of rocket without fuel [kg]

''' code from part 5 '''  

def enough_fuel(N_H2, fuel_m, t0, phi, home_pos, max_time, boosts=None):

    '''
    function for launching our rocket and checking if
    we have enough fuel left to perform the planned boosts
    during our travel
    '''

    #r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
    
    #particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
    #mean_f = f/steps                                        # the box force averaged over all time steps [N]
    #fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]
                    
    #thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]
    
    initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
    #mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

    ''' stored values from simulations '''

    thrust_f = 469328.7858317199
    mass_loss_rate = 37.85393604144184 
    
    ''' launching '''

    r0 = utils.m_to_AU(np.array([-np.sin(phi)*R_p[0], np.cos(phi)*R_p[0]])) + home_pos
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
        if type(boosts) != NoneType:
            tot_fuel_loss = 0
            for i in range(len(boosts[:, 0])):
                dv = np.linalg.norm(np.array([boosts[i, 1], boosts[i, 2]]))
                boosts[i, 3] = fuel_consumption(thrust_f, rocketm_afterlaunch - tot_fuel_loss, mass_loss_rate, dv)
                tot_fuel_loss += boosts[i, 3]
            
            rocketm_afterboosts = rocketm_afterlaunch - tot_fuel_loss
            if rocketm_afterboosts <= spacecraft_m:
                print(f"The rocket's mass is {rocketm_afterboosts:.2f} kg after completing the boosts! Bring more fuel with you")
            else:
                print(f"The rocket's mass is {rocketm_afterboosts:.2f} kg after completing the boosts! You're ready for travel :)")

                ''' converting to astronomical units '''
                
                rocketm_afterlaunch = rocketm_afterlaunch/m_sun
                boosts[:, 1] = utils.m_pr_s_to_AU_pr_yr(boosts[:, 1])
                boosts[:, 2] = utils.m_pr_s_to_AU_pr_yr(boosts[:, 2])
                boosts[:, 3] = boosts[:, 3]/m_sun
    
    return rocketm_afterlaunch, time_afterlaunch, pos_afterlaunch, vel_afterlaunch, boosts

rocketm_afterlaunch, time_afterlaunch, pos_afterlaunch, vel_afterlaunch, no_boosts = enough_fuel(N_H2, fuel_m, t0, phi, home_pos, max_time)

dt = times[1] - times[0]
time_stamp_afterlaunch = time_stamp + int((time_afterlaunch - t0)/dt)


''' manuel orientation '''

mission.take_picture(filename=r'/Users/paljettrosa/Documents/AST2000/sky_picture_part5.png', full_sky_image_path=r'/Users/paljettrosa/Documents/AST2000/himmelkule.npy')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sky_picture_part5.png')

dlambda_sun = np.array(mission.star_doppler_shifts_at_sun)
dlambda_sc = np.array(mission.measure_star_doppler_shifts())
phi_orient = np.array(mission.star_direction_angles)*np.pi/180                                                   
vxy = sc_velocity(dlambda_sun, dlambda_sc, phi_orient)

distances = mission.measure_distances()                     # the spacecraft's distance from each body in the solar system immediately after launch [AU]

''' adjusting the trilateration method so that it works better with this code '''

def trilateration(time_stamp, distances):
    planet_pos = exact_planet_positions[time_stamp]         # updating the planets' positions to where they are after launch [AU]
    
    r_Bl = planet_pos[1]                                    # Blossom's position relative our sun immediately after launch [AU]
    r_F = planet_pos[4]                                     # Flora's position relative our sun immediately after launch [AU]
    
    distance_Bl = distances[1]                              # the spacecraft's distance to Blossom [AU]
    distance_F = distances[4]                               # the spacecraft's distance to Flora [AU]
    distance_s = distances[-1]                              # the spacecraft's distance to our sun [AU]
    
    N = int(1e5)
    tol = 1e-4
    theta_array = np.linspace(0, 2*np.pi, N)
    
    for angle in theta_array:
        r_s = np.array([np.cos(angle), np.sin(angle)])*distance_s       # guess for the spacecraft's positional vector relative to our sun[AU]
        
        if abs(np.linalg.norm(r_s - r_Bl) - distance_Bl) < tol and abs(np.linalg.norm(r_s - r_F) - distance_F) < tol:
            break

    print(f"The spacecraft's position relative to the sun is x = {r_s[0]:g} AU, y = {r_s[1]:g} AU immediately after the launch")
    return r_s

r_s = trilateration(time_stamp_afterlaunch, distances)

mission.verify_manual_orientation(r_s, vxy, phi0)


''' interplanetary travel '''

travel = mission.begin_interplanetary_travel()
travel.look_in_direction_of_planet(0)
travel.take_picture('where_is_homeplanet.xml')
print('\n')
time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

travel.boost([-0.7, 0.3])
travel.coast(3)
travel.look_in_direction_of_planet(3)
travel.take_picture('checking_position1.xml')
travel.look_in_direction_of_motion()
travel.take_picture('checking_motion_direction.xml')
print('\n')
time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

travel.boost([2.55, -0.75])
travel.coast(2.1)
travel.look_in_direction_of_planet(3)
travel.take_picture('checking_position2.xml')
print('\n')
time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

travel.boost([-0.2, -0.2])
travel.coast(2) 
travel.take_picture('checking_position3.xml')
print('\n')
time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

''' stabilizing the orbit '''

travel.coast(0.0015)
travel.boost([(Buttercup_vel[0] - velocity[0])/2, 0])
print('\n')
time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

''' filming the orbit '''

travel.start_video()
travel.coast(0.02)
travel.finish_video('stabilized_orbit.xml')
print('\n')

''' final orientation '''

time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")






''' code for part 6 '''

'''
A. Enter a Low Orbit
'''
'''
Task 1 and 2
'''

travel.start_video()
for i in range(100):
    travel.coast(0.0005)
    time, position, velocity = travel.orient()
    time_stamp = int(time/dt)
    Buttercup_pos = r_Buttercup[time_stamp]
    Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
    distance = utils.AU_to_km(np.linalg.norm(position - Buttercup_pos))
    print(f"The current distance between the spacecraft and Buttercup is {distance:.2f} km")
    if i < 60:
        if distance >= 1e5:
            travel.boost([- (velocity[0] - Buttercup_vel[0])/18, 0])    
    if i == 60:
        travel.finish_video('lowering.xml')
    
    ''' filming the orbit to check if we've successfully lowered it '''

    if i == 80:
        travel.start_video()
travel.finish_video('lowered_orbit.xml')





'''
D. Scouting for Landing Sites
'''
'''
Task 1
'''
T = T_p[3]                                      # Buttercup's rotational period [s]
R = R_p[3]                                      # Buttercup's radius [m]

def new_coords(coords, t_elapsed):

    '''
    function for calculating the potential landing 
    sites' new coordinates as time passes by
    '''

    t_elapsed += (coords[-1, 3] - coords[:, 3])
    new_coords = coords
    deltaphi = 2*np.pi*t_elapsed/T              # the change in the landing sites' phi-coordinate during the elapsed time
    new_coords[:, 1] += deltaphi
    new_coords[:, 3] += t_elapsed
    return new_coords


''' 
Task 2 
'''

'''
After looking at videos of our planet, we can see that there are many potential landing sites on the surface. We
therefore choose to stay with the initial z-coordinate when we switch to three coordinates. This means that the
theta-coordinate is always pi/2, since all our potential landing sites will be along the equator of Buttercup, and 
its rotational axis is the z-axis. The rho-coordinate is also constant, since we approximate the planet as a perfect 
sphere. The rho-coordinate is therefore always R, where R is the radius of Buttercup. The phi-coordinate changes 
linearly, as Buttercup's rotational velocity is constant.
'''

final_m = travel.remaining_fuel_mass + mission.spacecraft_mass
travel.record_destination(3)
landing = mission.begin_landing_sequence()
landing.look_in_direction_of_planet(3)

landing.fall(0.00005*yr)
time, position, velocity = landing.orient()
coords = np.zeros((10, 4))            
for i in range(10):
    time, position, velocity = landing.orient()
    phi = np.arctan(position[1]/position[0])
    coords[i] = np.array([R, phi, np.pi/2, time])
    landing.take_picture(f'scouting{i+1}.xml')
    if i != 9:
        landing.fall(0.0001*yr)
