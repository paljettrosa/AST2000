#EGEN KODE
import numpy as np
from PIL import Image
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

AU = const.AU                               # astronomical unit of length [m]
M = const.m_sun                             # solar mass [kg]
yr = const.yr                               # one year [s]
AU_N = M*AU*yr**2                           # astronomical unit of force [N]

'''
B. Image Analysis
'''
def find_phi0(pic_name, width = 640, length = 480, theta0 = np.pi/2):
    pic = Image.open(pic_name)
    pixels = np.array(pic)
    center_pixels = pixels[(int(length/2)-2):(int(length/2)+2), (int(width/2)-2):(int(width/2)+2)]
    for i in range(360):
        ref_pic = Image.open(f'phi_{i}.0_deg.png')
        ref_pixels = np.array(ref_pic)
        center_ref_pixels = ref_pixels[(int(length/2)-2):(int(length/2)+2), (int(width/2)-2):(int(width/2)+2)]
        if np.all(center_pixels == center_ref_pixels):
            phi0 = i
            print(f"The angle that the spacecraft was facing when {pic_name} was taken was at phi = {phi0} degrees")
            break
    return phi0
    
phi0 = find_phi0('phi_42.0_deg.png')

'''
C. Doppler Shift Analysis
'''
'''
Task 1 and 2
'''
c = const.c                                                 # the speed of light [m/s]
lambda0 = mission.reference_wavelength                      # the wavelength of the spectral line of H-alpha at rest [nm]

def sc_velocity(dlambda_sun, dlambda_sc, phi):
    vr_sun = c*dlambda_sun/lambda0          # radial velocity of our sun relative to the stars [m/s]
    vr_sc = c*dlambda_sc/lambda0            # radial velocity of our spacecraft relative to the stars [m/s]
    
    '''defining a new coordinate system using the method introduced in the relevant mathematics'''
    
    u = np.zeros((2, 2))
    for i in range(2):
        u[i] = np.array([np.cos(phi[i]), np.sin(phi[i])])                               # defining the new coordinate vectors
    M = 1/np.sin(phi[1] - phi[0])*np.array([[np.sin(phi[1]), -np.sin(phi[0])],          # change of coordinates matrix 
                                            [-np.cos(phi[1]), np.cos(phi[0])]])
    
    vr = vr_sun - vr_sc                     # radial velocity of the spacecraft relative to our sun [m/s]
    vphi = vr*u                             # velocity of the spacecraft relative to our sun in (phi1, phi2)-coordinates [m/s]
    vxy = np.matmul(M, vphi)                # velocity of the spacecraft relative to our sun in (x, y)-coordinates [m/s]
    
    #RIKTIG MÅTE Å REGNE UT VPHI????? 
    #BARE RETURNERE VXY?????
    #TESTE FUNKSJONEN?????
    vxy = vxy/AU/yr                         # changing to astronomical units
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
def trilateration(time, distances):
    init_pos = np.transpose(system.initial_positions)       # planets' initial positions relative to our sun [AU]
    init_vel = np.transpose(system.initial_velocities)      # planets' initial velocities relative to our sun [AU/yr]
    planet_pos = init_pos + init_vel*time                   # updating the planets' positions to where they are after launch [AU]
    
    r_Bl = planet_pos[1]                                    # Blossom's position relative our sun immediately after launch [AU]
    r_F = planet_pos[4]                                     # Flora's position relative our sun immediately after launch [AU]
    
    distance_Bl = distances[1]                              # the spacecraft's distance to Blossom [AU]
    distance_F = distances[4]                               # the spacecraft's distance to Flora [AU]
    distance_s = distances[-1]                              # the spacecraft's distance to our sun [AU]
    
    N = 1e5
    tol = 1e-5
    theta_array = np.linspace(0, 2*np.pi, N)
    theta = 0
    
    for angle in theta_array:
        r_s = np.array([np.cos(angle), np.sin(angle)])*distance_s       # guess for the spacecraft's positional vector relative to our sun[AU]
        
        if np.linalg.norm(r_s - r_Bl) - distance_Bl < tol and np.linalg.norm(r_s - r_F) - distance_F < tol:
            theta = angle
            break
        
    x = r_s*np.cos(theta)                                   # the spacecraft's x-position relative to the sun immediately after launch [AU]
    y = r_s*np.sin(theta)                                   # the spacecraft's y-position relative to the sun immediately after launch [AU]
    print("The spacecraft's position relative to the sun is x = {x:g} AU, y = {y:g} AU immediately after the launch")
    
    r_s = np.array([x, y])
    return r_s

#FIKS VERIFY_LAUNCH_RESULT FØR VI FINNER DISTANCES, RIKTIG TID BRUKT???
#RETURNERE X,Y ISTEDET FOR R_S???
time = 491.67/yr                                            # time stamp when the launch was completed [yr]
distances = mission.measured_distances()                    # the spacecraft's distance from each body in the solar system immediately after launch [AU]
#REGN UT DISTANCES ISTEDET??? ELLER KAN VI BRUKE DETTE HER???
r_s = trilateration(time, distances)

'''
Manuel Orientation
'''
mission.take_picture()
mission.measure_star_doppler_shifts()
mission.measure_distances()

mission.verify_manual_orientation(r_s, vxy, 0.0)

#SKULLE VI GJETTE AT ANGLE ETTER LAUNCH VAR 0????
#ER DETTE GJORT RIKTIG???? HVA MENER DE MED STEP 2????

'''
1. The first step is to use the satellite’s onboard
equipement in order to gather the necessary data.
This is done using the following methods of your
SpaceMission instance: take picture (rotational
orientation), measure star doppler shifts (velocity) and measure distances (position).

2. The next step is analyse the data using your orientation software.

3. The final step is to use the
verify manual orientation method in order to verify your calculations have been done
correctly.
'''

