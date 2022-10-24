#EGEN KODE
import numpy as np
from PIL import Image
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

AU = const.AU                               # astronomical unit of length [m]
M = const.m_sun                             # solar mass [kg]
yr = const.yr                               # one year [s]
AU_N = M*AU*yr**2                           # astronomical unit of force [N]

init_pos = np.transpose(system.initial_positions)       # planets' initial positions relative to our sun [AU]
init_vel = np.transpose(system.initial_velocities)      # planets' initial velocities relative to our sun [AU/yr]
R = utils.km_to_AU(system.radii)                        # planets' radiis [AU]

A_box = L*L                                     # area of one gasbox [m**2]
A_spacecraft = mission.spacecraft_area          # area of our spacecraft's cross section [m**2]
N_box = int(A_spacecraft/A_box)                 # number of gasboxes   

spacecraft_m = mission.spacecraft_mass          # mass of rocket without fuel [kg]


'''
B. Image Analysis
'''
def find_phi0(pic_path, width = 640, length = 480, theta0 = np.pi/2):
    pic = Image.open(pic_path)
    pixels = np.array(pic)
    center_pixels = pixels[(int(length/2)-5):(int(length/2)+5), (int(width/2)-5):(int(width/2)+5)]
    for i in range(3591):
        ref_pic = Image.open(f'/Users/paljettrosa/Documents/AST2000/phi/phi_{i/10:.1f}_deg.png')
        ref_pixels = np.array(ref_pic)
        center_ref_pixels = ref_pixels[(int(length/2)-5):(int(length/2)+5), (int(width/2)-5):(int(width/2)+5)]
        if np.all(center_pixels == center_ref_pixels):
            phi0 = i/10
            print(f"The angle that the spacecraft was facing when the given picture was taken was at phi = {phi0} degrees")
            return phi0

phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sample0000.png')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sample0200.png')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sample0435.png')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sample0911.png')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sample1400.png')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sample1900.png')




'''
C. Doppler Shift Analysis
'''
'''
Task 1 and 2
'''
c = const.c_AU_pr_yr                                        # the speed of light [AU/yr]
lambda0 = mission.reference_wavelength                      # the wavelength of the spectral line of H-alpha at rest [nm]

def sc_velocity(dlambda_sun, dlambda_sc, phi):
    vr_sun = c*dlambda_sun/lambda0          # radial velocities of the stars relative to our sun [m/s]
    vr_sc = c*dlambda_sc/lambda0            # radial velocities of the stars relative to our spacecraft [m/s]
    v_phi = vr_sun - vr_sc                  # radial velocity of the spacecraft relative to our sun, in (phi1, phi2)-coordinates [m/s]

    M = 1/np.sin(phi[1] - phi[0])*np.array([[np.sin(phi[1]), -np.sin(phi[0])],          # change of coordinates matrix 
                                            [-np.cos(phi[1]), np.cos(phi[0])]])
    
    vxy = np.matmul(M, v_phi)               # velocity of the spacecraft relative to our sun in (x, y)-coordinates [m/s]

    print(f"The spacecraft's cartesian velocity relative to the sun is vx = {vxy[0]:.3g} AU/yr, vy = {vxy[1]:.3g} AU/yr")
    return vxy

'''
since our spacecraft's spectrograph measures dlambda = 0 for both radial 
velocities, our spacecraft does not have a radial velocity component in
respect to either of the reference stars
'''

dlambda_sun = np.array(mission.star_doppler_shifts_at_sun)  # Doppler shift of the stars relative to our sun [nm]
dlambda_sc = np.array([0.0, 0.0])                           # Doppler shift of the stars relative to the spacecraft [nm]
phi = np.array(mission.star_direction_angles)               # azimuthal angles of the stars relative to the solar system x-axis in degrees
phi = phi*np.pi/180                                         # converting to radians

vxy = sc_velocity(dlambda_sun, dlambda_sc, phi)




'''
D. Spacecraft Trilateration
'''
def trilateration(launch_duration, distances):
    planet_pos = init_pos + init_vel*launch_duration        # updating the planets' positions to where they are after launch [AU]
    
    r_Bl = planet_pos[1]                                    # Blossom's position relative our sun immediately after launch [AU]
    r_F = planet_pos[4]                                     # Flora's position relative our sun immediately after launch [AU]
    
    distance_Bl = distances[1]                              # the spacecraft's distance to Blossom [AU]
    distance_F = distances[4]                               # the spacecraft's distance to Flora [AU]
    distance_s = distances[-1]                              # the spacecraft's distance to our sun [AU]
    
    N = int(1e5)
    tol = 1e-5
    theta_array = np.linspace(0, 2*np.pi, N)
    
    for angle in theta_array:
        r_s = np.array([np.cos(angle), np.sin(angle)])*distance_s       # guess for the spacecraft's positional vector relative to our sun[AU]
        
        if np.linalg.norm(r_s - r_Bl) - distance_Bl < tol and np.linalg.norm(r_s - r_F) - distance_F < tol:
            break

    print(f"The spacecraft's position relative to the sun is x = {r_s[0]:g} AU, y = {r_s[1]:g} AU immediately after the launch")
    return r_s





''' 
Manuel Orientation
'''

''' launching at t0 = 0.0, from the equator '''

dt = 1                                  # time step length for launch [s]
max_time = 20*60                        # maximum launch duration [s]

N_H2 = 5*10**6
fuel_m = 5*10**4

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
    
particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                                        # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]
                    
thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]

initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

r0 = np.array([R[0], 0.0]) + init_pos[0]
mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = max_time, 
                              launch_position = r0, 
                              time_of_launch = 0.0)

mission.launch_rocket()
fuel_consumed, time_afterlaunch, pos_afterlaunch, vel_afterlaunch = shortcut.get_launch_results()
mission.verify_launch_result(pos_afterlaunch)

''' finding the angle of the spacecraft immediately after launch '''

mission.take_picture(filename=r'/Users/paljettrosa/Documents/AST2000/sky_picture.png', full_sky_image_path=r'/Users/paljettrosa/Documents/AST2000/himmelkule.npy')
phi0 = find_phi0(r'/Users/paljettrosa/Documents/AST2000/sky_picture.png')

''' finding the velocity of the spacecraft immediately after launch '''

dlambda_sc = np.array(mission.measure_star_doppler_shifts())
vxy = sc_velocity(dlambda_sun, dlambda_sc, phi)

''' finding the position of the spacecraft immediately after launch'''

launch_duration = time_afterlaunch/yr                       # time stamp when the launch was completed [yr]
distances = mission.measure_distances()                     # the spacecraft's distance from each body in the solar system immediately after launch [AU]
r_s = trilateration(launch_duration, distances)

''' verification '''

mission.verify_manual_orientation(r_s, vxy, phi0)

'''
RESULTS:

        TESTING THE SOFTWARE:
    The angle that the spacecraft was facing when the given picture was taken was at phi = 0.0 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 20.0 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 43.5 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 91.1 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 140.0 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 190.0 degrees
    The spacecraft's cartesian velocity relative to the sun is vx = 0.471 AU/yr, vy = 0.141 AU/yr

        LAUNCH:
    Rocket was moved up by 4.50388e-06 m to stand on planet surface.
    New launch parameters set.
    Launch completed, reached escape velocity in 524.92 s.
    Your spacecraft position was satisfyingly calculated. Well done!
    *** Achievement unlocked: No free launch! ***

        USING THE SOFTWARE AFTER LAUNCH:
    Picture written to /Users/paljettrosa/Documents/AST2000/sky_picture.png.
    The angle that the spacecraft was facing when the given picture was taken was at phi = 161.0 degrees
    The spacecraft's cartesian velocity relative to the sun is vx = 2.16 AU/yr, vy = 5.66 AU/yr
    The spacecraft's position relative to the sun is x = 3.54097 AU, y = 0.000222488 AU immediately after the launch

        MANUAL ORIENTATION:
    Pointing angle after launch correctly calculated. Well done!
    Velocity after launch correctly calculated. Well done!
    Position after launch correctly calculated. Well done!
    Your manually inferred orientation was satisfyingly calculated. Well done!
    *** Achievement unlocked: Well-oriented! ***
'''
