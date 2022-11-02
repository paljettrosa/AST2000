#EGEN KODE
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts 
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [10978]) 

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = 4*np.pi**2                                  # gravitation constant [AU**3yr**(-2)M**(-1)]
G_SI = const.G                                  # gravitation constant in SI-units [m**3s**(-2)kg**(-1)]
m_sun = const.m_sun                             # solar mass [kg]
AU = const.AU                                   # one astronomical unit [m]
yr = const.yr                                   # one year [s]

M_p = system.masses                             # planets' masses [M]
M_s = system.star_mass                          # sun's mass [M]
M_p_SI = M_p*m_sun                              # planets' masses in SI-units [kg]
M_s_SI = M_s*m_sun                              # sun's mass in SI-units [kg]
R_p = system.radii*1e3                          # planets' radiis [m]
a_SI = utils.AU_to_m(system.semi_major_axes)    # each planet's semi-major axis in SI-units [m]
r_s = np.array([0.0, 0.0])                      # sun's position [AU]

A_box = L*L                                     # area of one gasbox [m**2]
A_spacecraft = mission.spacecraft_area          # area of our spacecraft's cross section [m**2]
N_box = int(A_spacecraft/A_box)                 # number of gasboxes   

spacecraft_m = mission.spacecraft_mass          # mass of rocket without fuel [kg]


@jit(nopython = True)
def fuel_consumption(thrust_f, rocket_m, mass_loss_rate, delta_v):
    
    '''
    we assume that the amount of fuel that the rocket loses during the speed boost
    is so minimal that we can define the rocket's acceleration as the total thrust force
    divided by it's total mass before the boost
    '''
    
    a = thrust_f/rocket_m                                   # the rocket's acceleration [m/s**2]
    delta_t = delta_v/a                                     # time spent accelerating the rocket [s]
    tot_fuel_loss = abs(delta_t*mass_loss_rate)             # total fuel loss [kg]
    
    return tot_fuel_loss


def enough_fuel(N_H2, fuel_m, t0, phi, home_pos, max_time, boosts=None):
    '''
    r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
    
    particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
    mean_f = f/steps                                        # the box force averaged over all time steps [N]
    fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]
                       
    thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]
    
    initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
    mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]
    
    
    #TODO fjern kommentering og endre tilbake til thrust_f og mass_loss_rate
    print(thrust_f)
    print(mass_loss_rate)
    print(fuel_loss_s)
    "TODO dette er med T=4000K
    thrust_f = 612396.3173827773
    mass_loss_rate = 129.7720234644266
    fuel_loss_s = 8.110751466526664e-12
    '''

    ''' adjusting our mass loss rate so that we actually can boost our rocket '''
    
    initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
    thrust_f = 469545.20630358486
    mass_loss_rate = 113.65783969805446/3
    thrust_f = 469328.7858317199
    mass_loss_rate = 113.56180812432552/3 #TODO del heller på 3 ovenfor
    #mass_loss_rate = mass_loss_rate/3
    
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




'''
A. Simulating the Spacecraft's Trajectory
'''

def trajectory(time_afterlaunch, time_stamp_afterlaunch, dt, r0, v0, m, r_p, p_idx, boosts=None):          
    r_p = r_p[time_stamp_afterlaunch:]
    r = np.zeros_like(r_p[:, 0])
    v = np.zeros_like(r_p[:, 0])
    r[0] = r0
    v[0] = v0

    distance = np.zeros(len(planets))
    l = np.zeros(len(planets))
    bool_val = True
    
    for i in range(len(r_p[:, 0, 0]) - 1):
        
        ''' checking if it's time for a speed boost '''

        if type(boosts) != NoneType:
            for ti, dvx, dvy, dm in boosts:              
                if i == ti:                  
                    v[i] = v[i] + [dvx, dvy]    # updating the rocket's velocity
                    m = m - dm                  # updating the rocket's mass
        
        pos = r[i] - r_s
        g = - G*M_s/np.linalg.norm(pos)**3*(pos)
        for j in range(len(planets)):
            pos = r[i] - r_p[i, j]
            g -= G*M_p[j]/np.linalg.norm(pos)**3*(pos)
        
        v[i+1] = v[i] + g*dt/2
        r[i+1] = r[i] + v[i+1]*dt

        pos = r[i+1] - r_s
        g = - G*M_s/np.linalg.norm(pos)**3*(pos)
        for j in range(len(planets)):
            pos = r[i+1] - r_p[i+1, j]
            g -= G*M_p[j]/np.linalg.norm(pos)**3*(pos)

        v[i+1] = v[i+1] + g*dt/2
        
        ''' checking if we've reached our destination '''
        
        for j in range(len(planets)):
            distance[j] = np.linalg.norm(r[i+1] - r_p[i+1, j])                  # our spacecraft's distance from its destination
            l[j] = np.linalg.norm(r[i+1])*np.sqrt(M_p[j]/(10*M_s))              # the maximal distance between the spacecraft and its destination in
                                                                                # order for it to be able to to perform an orbital injection manouver
            if distance[j] <= l[j]:
                if j == p_idx:
                    #TODO krav oppfylt for at vi kan starte orbital injection?
                    print(f"You're close enough to {planets[p_idx, 0]} to perform an orbital injection! :)")
                    print(f"The spacecraft's distance to {planets[p_idx, 0]} is {distance[p_idx]:.3g} AU")
                    print(f"The spacecraft is close enough to perform an orbital injection {(i+1)*dt:.3f} years after launch")
                    final_time = time_afterlaunch + (i+1)*dt
                    final_pos = r[i+1]
                    final_vel = v[i+1]
                    final_mass = m
                    r = r[:i+1]
                    v = v[:i+1]
                    bool_val = False
                    break

                elif j != 0 and j != p_idx:
                    print(f"You're going into orbit around {planets[j, 0]}. This is the wrong planet! Adjust your boosts")
                    final_time = time_afterlaunch + (i+1)*dt
                    final_pos = r[i+1]
                    final_vel = v[i+1]
                    final_mass = m
                    r = r[:i+1]
                    v = v[:i+1]
                    bool_val = False
                    break

                else:
                    continue
        
        if bool_val == False:
            break

        if i*dt >= 7:
            print("You've drifted off! Adjust your boosts")
            final_time = time_afterlaunch + (i+1)*dt
            final_pos = r[i+1]
            final_vel = v[i+1]
            final_mass = m
            r = r[:i+1]
            v = v[:i+1]
            break
    
    return r, v, final_time, final_pos, final_vel, final_mass, distance[p_idx] #TODO, riktig distance som de spør om i 5D?




'''
B. Plan your Journey
'''   

''' 
we want to take use of the Hohmann transfer orbit method in order to go into orbit around Buttercup
this method assumes circular planetary orbits, where the radius of the orbits is the semi-major axes 
'''

def Hohmann(N, R1, R2, r1, r2):
    alpha = np.pi*(1 - 1/(2*np.sqrt(2))*np.sqrt((R1/R2 + 1)**3))        # optimal angular alignment

    tol = 1e-3
    for i in range(N - 1):
        angle1 = np.arccos(r1[i, 0]/np.linalg.norm(r1[i]))*np.sign(r1[i, 1])
        angle2 = np.arccos(r2[i, 0]/np.linalg.norm(r2[i]))*np.sign(r2[i, 1])
        if abs(abs(angle1 - angle2) - alpha) <= tol:
            time_stamp = i
            break

    my = G_SI*M_s_SI
    deltav1 = np.sqrt(my/R1)*(np.sqrt(2*R2/(R1 + R2)) - 1)              # TODO riktig? speed boost required for the Hohmann transfer [m/s]
    deltav2 = np.sqrt(my/R2)*(1 - np.sqrt(2*R1/(R1 + R2)))              # TODO riktig? speed boost required to enter a new circular orbit [m/s]

    return time_stamp, angle1, deltav1, deltav2

R_home = a_SI[0]                                                        # radius of the home planet's orbit
R_Buttercup = a_SI[3]                                                   # radius of the Buttercup's orbit

f = np.load(r'/Users/paljettrosa/Documents/AST2000/planet_trajectories.npz')
times = f['times']
exact_planet_positions = np.einsum("ijk->kji", f['planet_positions'])

r_home = exact_planet_positions[:, 0]
r_Buttercup = exact_planet_positions[:, 3]

N = 10**5           # amount of time steps

time_stamp, phi, deltav1, deltav2 = Hohmann(N, R_home, R_Buttercup, r_home*AU, r_Buttercup*AU)

''' 
we now need to find our velocity after launch so that we can simulate the 
spacecraft's trajectory if we let it drift for a certain time period after launch. 
we also want to compare the trajectory to Buttercups trajectory during 
the same time period.
'''

N_H2 = 6*10**6                          # amount of H2-molecules
fuel_m = 4.5*10**4                      # amount of fuel [kg]
max_time = 20*60                        # maximum launch duration [s]

t0 = times[time_stamp]
home_pos = exact_planet_positions[time_stamp, 0]



def main():
    rocketm_afterlaunch, time_afterlaunch, pos_afterlaunch, vel_afterlaunch, no_boosts = enough_fuel(N_H2, fuel_m, t0, phi, home_pos, max_time)

    dt = times[1] - times[0]
    time_stamp_afterlaunch = time_stamp + int((time_afterlaunch - t0)/dt)
    r, v, final_time, final_pos, final_vel, final_mass, distance = trajectory(time_afterlaunch, time_stamp_afterlaunch, dt, pos_afterlaunch, vel_afterlaunch, rocketm_afterlaunch, exact_planet_positions, 3)

    time_stamp_aftertrajectory = int(final_time/dt)
    Buttercup_trajectory = r_Buttercup[time_stamp_afterlaunch:time_stamp_aftertrajectory]

    plt.plot(r_home[:, 0], r_home[:, 1], color = 'crimson', label = "home planet's orbit")
    plt.plot(Buttercup_trajectory[:, 0], Buttercup_trajectory[:, 1], color = 'olivedrab', label = "Buttercup's trajectory")
    plt.plot(r[:, 0], r[:, 1], color = 'gold', label = "spacecraft's trajectory")
    plt.legend()
    plt.axis('equal')
    plt.title(f"The spacecraft's and Buttercup's trajectories the first {final_time - t0:.2f}\nyears after launch, when we let the spacecraft only drift")
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/trajectory_drift.pdf')


    '''
    we now need to implement boosts in order to aim the spacecraft towards 
    Buttercup. to do this, we've adjusted our trajectory function from A so 
    that it can take in arrays containing time stamps when we want to boost 
    our speed, and the difference in both velocity components. we also have to 
    check if our spacecraft has enough fuel left after the boost is completed
    '''

    #deltav1_vec = np.array([deltav1*np.cos(phi), deltav1*np.sin(phi)]) TODO skal disse hastighetene brukes?

    boosts = np.array([[int(4/dt), -3000, 3000, 0.0],
                    [int(4.5/dt), -1550, 2200, 0.0]])

    rocketm_afterlaunch, time_afterlaunch, pos_afterlaunch, vel_afterlaunch, boosts = enough_fuel(N_H2, fuel_m, t0, phi, home_pos, max_time, boosts)

    time_stamp_afterlaunch = time_stamp + int((time_afterlaunch - t0)/dt)
    print(time_stamp_afterlaunch) 

    r, v, final_time, final_pos, final_vel, final_mass, distance = trajectory(time_afterlaunch, time_stamp, dt, pos_afterlaunch, vel_afterlaunch, rocketm_afterlaunch, exact_planet_positions, 3, boosts)

    time_stamp_aftertrajectory = int(final_time/dt)
    Buttercup_trajectory = r_Buttercup[time_stamp_afterlaunch:time_stamp_aftertrajectory]

    plt.plot(r_home[:, 0], r_home[:, 1], color = 'crimson', label = "home planet's orbit")
    plt.plot(Buttercup_trajectory[:, 0], Buttercup_trajectory[:, 1], color = 'olivedrab', label = "Buttercup's trajectory")
    plt.plot(r[:, 0], r[:, 1], color = 'gold', label = "spacecraft's trajectory")
    plt.legend()
    plt.axis('equal')
    plt.title(f"The spacecraft's and Buttercup's trajectories the first\n{final_time - t0:.3f} years after launch, with boosts")
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/trajectory_wboosts.pdf')

if __name__ == '__main__':
    main()




'''
RESULTS:

        LAUNCH:
    Rocket was moved up by 1.79848e-05 m to stand on planet surface.
    New launch parameters set.
    Note: Existing launch results were cleared.
    Launch completed, reached escape velocity in 896.42 s.
    Your spacecraft position was satisfyingly calculated. Well done!
    *** Achievement unlocked: No free launch! ***


        SIMULATED ORBITAL INJECTION WITHOUT BOOSTS:
    You've drifted off! Adjust your boosts


        CHECKING SPEED BOOSTS:
    The rocket's mass is 6266.29 kg after completing the boosts! You're ready for travel :)
    [[ 1.02660000e+04 -6.32835585e-01  6.32835585e-01  2.09378692e-27]
     [ 1.15490000e+04 -3.26965052e-01  4.64079429e-01  8.73654063e-28]]   


        SIMULATED ORBITAL INJECTION WITH BOOSTS:
    You're close enough to Buttercup to perform an orbital injection! :)
    The spacecraft's distance to Buttercup is 0.00341 AU
    The spacecraft is close enough to perform an orbital injection 5.008 years after launch
'''