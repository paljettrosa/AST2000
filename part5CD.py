#EGEN KODE
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
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)
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

rocketm_afterlaunch, time_afterlaunch, pos_afterlaunch, vel_afterlaunch, no_boosts = enough_fuel(N_H2, fuel_m, t0, phi, home_pos, max_time)

dt = times[1] - times[0]
time_stamp_afterlaunch = time_stamp + int((time_afterlaunch - t0)/dt)



'''
C. Sending the Spacecraft
'''

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
travel.take_picture(r'/Users/paljettrosa/Documents/AST2000/MCast/data/where_is_homeplanet.xml')
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
travel.take_picture(r'/Users/paljettrosa/Documents/AST2000/MCast/data/checking_position1.xml')
travel.look_in_direction_of_motion()
travel.take_picture(r'/Users/paljettrosa/Documents/AST2000/MCast/data/checking_motion_direction.xml')
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
travel.take_picture(r'/Users/paljettrosa/Documents/AST2000/MCast/data/checking_position2.xml')
print('\n')
time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

travel.boost([-0.2, -0.2])
travel.coast(2) 
travel.take_picture(r'/Users/paljettrosa/Documents/AST2000/MCast/data/checking_position3.xml')
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
travel.finish_video(r'/Users/paljettrosa/Documents/AST2000/MCast/data/stabilized_orbit.xml')
print('\n')

''' final orientation '''

time, position, velocity = travel.orient()
time_stamp = int(time/dt)
Buttercup_pos = r_Buttercup[time_stamp]
Buttercup_vel = (r_Buttercup[time_stamp + 1] - r_Buttercup[time_stamp])/dt
print(f"Buttercup's position: ({Buttercup_pos[0]:.5f}, {Buttercup_pos[1]:.5f}) AU")
print(f"Buttercup's velocity: ({Buttercup_vel[0]:.5f}, {Buttercup_vel[1]:.5f}) AU/yr\n")

#TODO putt inn i en funksjon og gjør det mer sleek?
#TODO lag gif av movie med metode fra del 4





def main():
    '''
    D. Orbit Stability
    '''

    def orbit_stability(N, dt, x0, y0, vx0, vy0, M, R):
        xy = np.zeros((N, 2))
        vxy = np.zeros((N, 2))
        r = np.zeros(N)   
        vr = np.zeros(N)  
        vtheta = np.zeros(N)

        xy[0] = np.array([x0, y0])
        vxy[0] = np.array([vx0, vy0])
        r[0] = np.linalg.norm(xy)
        vr[0] = (x0*vx0 + y0*vy0)/r[0]
        vtheta[0] = (x0*vy0 - y0*vx0)/r[0]**2

        a = np.zeros(3)
        b = np.zeros(3)
        e = np.zeros(3)
        apoapsis = np.zeros(3)
        periapsis = np.zeros(3)
        P = np.zeros(3)
        
        distance = r[0]
        apoapsis[0] = distance                                            
        periapsis[0] = distance
        P[0] = 0
        
        orbits = 0
        for i in range(N - 1):
            g = - G_SI*M/r[i]**3*xy[i]                      # the rocket's acceleration at current time step [m/s**2]   
            vxy[i+1] = vxy[i] + g*dt/2                                            
            xy[i+1] = xy[i] + vxy[i+1]*dt 
            r[i+1] = np.linalg.norm(xy[i+1])  

            g = - G_SI*M/r[i+1]**3*xy[i+1]                  # the rocket's acceleration at current time step [m/s**2]
            vxy[i+1] = vxy[i+1] + g*dt/2 
            vr[i+1] = (xy[i+1, 0]*vxy[i+1, 0] + xy[i+1, 1]*vxy[i+1, 1])/r[i+1]
            vtheta[i+1] = (xy[i+1, 0]*vxy[i+1, 1] - xy[i+1, 1]*vxy[i+1, 0])/r[i+1]**2

            tol = (r[1] - r[0])/4
            distance = r[i+1]                               # the current distance from the planet's center of mass [m]
            
            ''' updating our approximation of the apoapsis and the periapsis '''
            
            if distance >= apoapsis[orbits]:
                apoapsis[orbits] = distance
            
            if distance <= periapsis[orbits]:
                periapsis[orbits] = distance

            ''' checking if we're drifting off or colliding with the planet '''

            if abs(distance - r[0]) > r[0]:
                print(f'The spacecraft drifts off after travelling for {(i+1)*dt*1e-3:.3f} years!')
                break

            if abs(distance) < R:
                print(f'The spacecraft collides with the planet after travelling for {(i+1)*dt*1e-3:.3f} years!')
                break
            
            ''' checking if we've done a full orbit '''
            
            if i+1 > P[orbits]/dt*orbits + 100:
                if abs(xy[i+1, 0] - xy[0, 0]) <= tol and abs(xy[i+1, 1] - xy[0, 1]) <= tol:
                    a[orbits] = (apoapsis[orbits] + periapsis[orbits])/2
                    b[orbits] = np.sqrt(apoapsis[orbits]*periapsis[orbits])
                    e[orbits] = np.sqrt(1 - b[orbits]**2/a[orbits]**2)
                    P[orbits] = 2*np.pi*np.sqrt(a[orbits]**3/(G_SI*M))
                    
                    print(f'Approximations after {orbits + 1} orbit(s):')
                    print(f'    Semi-major axis:   {a[orbits]*1e-3:.2f} km')
                    print(f'    Semi-minor axis:   {b[orbits]*1e-3:.2f} km')
                    print(f'    Eccentricity:      {e[orbits]:.2f}')
                    print(f'    Apoapsis:          {apoapsis[orbits]*1e-3:.2f} km')
                    print(f'    Periapsis:         {periapsis[orbits]*1e-3:.2f} km')
                    print(f'    Revolution period: {P[orbits]*12/yr:.2f} months')

                    orbits = orbits + 1 
                    if orbits < 3:
                        apoapsis[orbits] = r[0]                                            
                        periapsis[orbits] = r[0]
                    
            if orbits == 3:
                xy = xy[:i+1]
                vxy = vxy[:i+1]
                r = r[:i+1]
                vr = vr[:i+1]
                vtheta = vtheta[:i+1]
                break                    

        return xy, vxy, r, vr, vtheta           


    ''' we use SI-units to avoid round-off errors '''

    x0 = utils.AU_to_m(position[0]) - utils.AU_to_m(Buttercup_pos[0])
    y0 = utils.AU_to_m(position[1]) - utils.AU_to_m(Buttercup_pos[1])

    vx0 = utils.AU_pr_yr_to_m_pr_s(velocity[0]) - utils.AU_pr_yr_to_m_pr_s(Buttercup_vel[0])
    vy0 = utils.AU_pr_yr_to_m_pr_s(velocity[1]) - utils.AU_pr_yr_to_m_pr_s(Buttercup_vel[1])

    N = 10**7
    dt = 10
    xy, vxy, r, vr, vtheta = orbit_stability(N, dt, x0, y0, vx0, vy0, M_p_SI[3], R_p[3])

    plt.plot(xy[:, 0], xy[:, 1], color = 'gold', label = 'spacecraft')
    plt.scatter(0, 0, color = 'crimson', label = 'Buttercup')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('The 3 first orbits around Buttercup')
    plt.axis('equal')
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/spacecraftorbits_3yrs.pdf')

    t = np.linspace(0, len(r)*dt, len(r))
    plt.plot(t, vr, color = 'slateblue')
    plt.xlabel('t [s]')
    plt.ylabel('radial velocity [m/s]')
    plt.title("The spacecraft's radial velocity during\nthe 3 first orbits around Buttercup")
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/radialvelocity_3yrs.pdf')

    plt.plot(t, vtheta, color = 'mediumvioletred')
    plt.xlabel('t[s]')
    plt.ylabel('angular velocity [s^-1]')
    plt.title("The spacecraft's angular velocity during\nthe 3 first orbits around Buttercup")
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/angularvelocity_3yrs.pdf')

if '__name__' == '__main__':
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


        MANUAL ORIENTATION:
    Picture written to /Users/paljettrosa/Documents/AST2000/sky_picture_part5.png.
    The angle that the spacecraft was facing when the given picture was taken was at phi = 161.0 degrees
    The spacecraft's cartesian velocity relative to the sun is vx = -5.75 AU/yr, vy = 4.91 AU/yr
    The spacecraft's position relative to the sun is x = 2.33364 AU, y = 2.64536 AU immediately after the launch
    Pointing angle after launch correctly calculated. Well done!
    Velocity after launch correctly calculated. Well done!
    Position after launch correctly calculated. Well done!
    Your manually inferred orientation was satisfyingly calculated. Well done!
    *** Achievement unlocked: Well-oriented! ***


        INTERPLANETARY TRAVEL:
    Camera pointing towards planet 0.
    XML file /Users/paljettrosa/Documents/MCast/data/where_is_homeplanet.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Picture saved to /Users/paljettrosa/Documents/MCast/data/where_is_homeplanet.xml.

    Performed automatic orientation:
    Time: 0.536142 yr
    Position: (2.33353, 2.64545) AU
    Velocity: (-5.75403, 4.90902) AU/yr
    Buttercup's position: (-7.99632, 8.28334) AU
    Buttercup's velocity: (-2.22435, -2.19059) AU/yr

    Spacecraft boosted with delta-v (-0.7, 0.3) AU/yr (2743.91 kg of fuel was used).
    Spacecraft coasted for 3 yr.
    Camera pointing towards planet 3.
    XML file /Users/paljettrosa/Documents/MCast/data/checking_position1.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Picture saved to /Users/paljettrosa/Documents/MCast/data/checking_position1.xml.
    Camera pointing towards direction of motion.
    XML file /Users/paljettrosa/Documents/MCast/data/checking_motion_direction.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Picture saved to /Users/paljettrosa/Documents/MCast/data/checking_motion_direction.xml.

    Performed automatic orientation:
    Time: 3.53614 yr
    Position: (-11.4775, -0.302878) AU
    Velocity: (-2.47665, -2.35382) AU/yr
    Buttercup's position: (-11.44453, -0.18098) AU
    Buttercup's velocity: (0.06111, -3.13955) AU/yr

    Spacecraft boosted with delta-v (2.55, -0.75) AU/yr (4749.61 kg of fuel was used).
    Spacecraft coasted for 2.1 yr.
    Camera pointing towards planet 3.
    XML file /Users/paljettrosa/Documents/MCast/data/checking_position2.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Picture saved to /Users/paljettrosa/Documents/MCast/data/checking_position2.xml.

    Performed automatic orientation:
    Time: 5.63614 yr
    Position: (-9.49454, -6.38727) AU
    Velocity: (1.94271, -2.5088) AU/yr
    Buttercup's position: (-9.49213, -6.38951) AU
    Buttercup's velocity: (1.74761, -2.61009) AU/yr

    Spacecraft boosted with delta-v (-0.2, -0.2) AU/yr (456.089 kg of fuel was used).
    Spacecraft coasted for 2 yr.
    XML file /Users/paljettrosa/Documents/MCast/data/checking_position3.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Picture saved to /Users/paljettrosa/Documents/MCast/data/checking_position3.xml.

    Performed automatic orientation:
    Time: 7.63614 yr
    Position: (-4.78918, -10.4285) AU
    Velocity: (3.36664, -1.36889) AU/yr
    Buttercup's position: (-4.78894, -10.42786) AU
    Buttercup's velocity: (2.83649, -1.32874) AU/yr

    Spacecraft coasted for 0.0015 yr.
    Spacecraft boosted with delta-v (-0.265076, 0) AU/yr (388.105 kg of fuel was used).

    Performed automatic orientation:
    Time: 7.63764 yr
    Position: (-4.78412, -10.4302) AU
    Velocity: (2.9744, -0.892818) AU/yr
    Buttercup's position: (-4.78562, -10.42942) AU
    Buttercup's velocity: (2.83690, -1.32784) AU/yr

    Video recording started.
    Spacecraft coasted for 0.2 yr.
    XML file /Users/paljettrosa/Documents/MCast/data/checking_orbit_video.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Video with 1000 frames saved to /Users/paljettrosa/Documents/MCast/data/checking_orbit_video.xml.

    Performed automatic orientation:
    Time: 7.83764 yr
    Position: (-4.21046, -10.6806) AU
    Velocity: (3.2668, -1.15682) AU/yr
    Buttercup's position: (-4.21070, -10.67978) AU
    Buttercup's velocity: (2.90347, -1.17161) AU/yr


        ORBIT STABILITY:
    Approximations after 1 orbit(s):
        Semi-major axis:   126038.03 km
        Semi-minor axis:   121864.01 km
        Eccentricity:      0.26
        Apoapsis:          158205.50 km
        Periapsis:         93870.55 km
        Revolution period: 0.18 months
    Approximations after 2 orbit(s):
        Semi-major axis:   126038.03 km
        Semi-minor axis:   121864.01 km
        Eccentricity:      0.26
        Apoapsis:          158205.50 km
        Periapsis:         93870.55 km
        Revolution period: 0.18 months
    Approximations after 3 orbit(s):
        Semi-major axis:   126038.03 km
        Semi-minor axis:   121864.01 km
        Eccentricity:      0.26
        Apoapsis:          158205.50 km
        Periapsis:         93870.55 km
        Revolution period: 0.18 months
'''