#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
from PIL import Image
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps
from part3C import rocket_launch_generalized

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)

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

box_A = L*L                                     # area of one gasbox [m**2]
spacecraft_A = mission.spacecraft_area          # area of our spacecraft's cross section [m**2]
N_box = int(spacecraft_A/box_A)                 # number of gasboxes   
spacecraft_m = mission.spacecraft_mass          # mass of rocket without fuel [kg]

c = const.c_AU_pr_yr                            # the speed of light [AU/yr]
lambda0 = mission.reference_wavelength          # the wavelength of the spectral line of H-alpha at rest [nm]

f = np.load('planet_trajectories.npz')
times = f['times']
dt_p = times[1]
r_p = np.einsum("ijk->kji", f['planet_positions'])


def find_phi0(filename, width = 640, length = 480):

    '''
    function for finding the spacecraft's angular orientation
    immediately after launch by comparing a picture taken
    to our 3600 flattened pictures
    '''

    pic = Image.open(filename)
    pixels = np.array(pic)
    center_pixels = pixels[(int(length/2)-5):(int(length/2)+5), (int(width/2)-5):(int(width/2)+5)]
    for i in range(3600):
        ref_pic = Image.open(f'phi_{i/10:.1f}_deg.png')
        ref_pixels = np.array(ref_pic)
        center_ref_pixels = ref_pixels[(int(length/2)-5):(int(length/2)+5), (int(width/2)-5):(int(width/2)+5)]
        if np.all(center_pixels == center_ref_pixels):
            phi0 = i/10
            print(f"The angle that the spacecraft was facing when the given picture was taken was at phi = {phi0} degrees")
            return phi0


def sc_velocity(dlambda_sun, dlambda_sc, phi):

    '''
    function for finding the spacecraft's (x, y) velocity
    relative to the sun immediately after launch
    '''

    vr_sun = c*dlambda_sun/lambda0          # radial velocities of the stars relative to our sun [AU/yr]
    vr_sc = c*dlambda_sc/lambda0            # radial velocities of the stars relative to our spacecraft [AU/yr]
    v_phi = vr_sun - vr_sc                  # radial velocity of the spacecraft relative to our sun, in (ϕ1, ϕ2)-coordinates [AU/yr]

    M = 1/np.sin(phi[1] - phi[0])*np.array([[np.sin(phi[1]), -np.sin(phi[0])],          # change of coordinates matrix 
                                            [-np.cos(phi[1]), np.cos(phi[0])]])
    
    vxy = np.matmul(M, v_phi)               # velocity of the spacecraft relative to our sun in (x, y)-coordinates [AU/yr]

    print(f"The spacecraft's (ϕ1, ϕ2)-velocity relative to our sun is vϕ1 = {v_phi[0]:.3f} AU/yr, vϕ2 = {v_phi[1]:.3f} AU/yr")
    print(f"The spacecraft's (x, y)-velocity relative to our sun is vx = {vxy[0]:.3f} AU/yr, vy = {vxy[1]:.3f} AU/yr")
    return vxy


def trilateration(t0, launch_duration, distances):

    '''
    function for finding the spacecraft's positional coordinates
    relative to the sun immediately after launch
    '''

    planet_pos = r_p[int((t0+launch_duration)/dt_p)]        # updating the planets' positions to where they are after launch [AU]
    
    r_Bl = planet_pos[1]                                    # Blossom's position relative our sun immediately after launch [AU]
    r_F = planet_pos[4]                                     # Flora's position relative our sun immediately after launch [AU]

    print("Planets' positions relative the sun after launch:")
    print(f"    Blossom: [{r_Bl[0]:.3f}, {r_Bl[1]:.3f}] AU")
    print(f"      Flora: [{r_F[0]:.3f}, {r_F[1]:.3f}] AU")
    
    distance_Bl = distances[1]                              # the spacecraft's distance to Blossom [AU]
    distance_F = distances[4]                               # the spacecraft's distance to Flora [AU]
    distance_s = distances[-1]                              # the spacecraft's distance to our sun [AU]

    print("Distances after launch:")
    print(f"    Blossom: {distance_Bl:.3f} AU")
    print(f"      Flora: {distance_F:.3f} AU")
    print(f"        sun: {distance_s:.3f} AU")
    
    N = int(1e5)
    tol = 1e-4
    theta_array = np.linspace(0, 2*np.pi, N)
    
    for angle in theta_array:
        r_s = np.array([np.cos(angle), np.sin(angle)])*distance_s       # guess for the spacecraft's positional vector relative to our sun[AU]
        
        if abs(np.linalg.norm(r_s - r_Bl) - distance_Bl) < tol and abs(np.linalg.norm(r_s - r_F) - distance_F) < tol:
            break

    print(f"The spacecraft's position relative to the sun is x = {r_s[0]:g} AU, y = {r_s[1]:g} AU immediately after the launch")
    return r_s




def main():
    '''
    B. Image Analysis
    '''

    phi0 = find_phi0('sample0000.png')
    phi0 = find_phi0('sample0200.png')
    phi0 = find_phi0('sample0435.png')
    phi0 = find_phi0('sample0911.png')
    phi0 = find_phi0('sample1400.png')
    phi0 = find_phi0('sample1900.png')


    '''
    C. Doppler Shift Analysis
    '''

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
    Manuel Orientation
    '''

    ''' launching at t0 = 0, from the far east side of the equator to test our software '''

    dt = 1                                  # time step length for launch [s]
    max_time = 20*60                        # maximum launch duration [s]

    N_H2 = 6*10**6
    fuel_m = 4.5*10**4

    r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)
        
    particles_s = exiting/time                              # the number of particles exiting per second [s**(-1)]
    mean_f = f/steps                                        # the box force averaged over all time steps [N]
    fuel_loss_s = particles_s*m_H2                          # the total fuel loss per second [kg/s]
                        
    thrust_f = N_box*mean_f                                 # the combustion chamber's total thrust force [N]

    initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
    mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

    phi_launch = 0                          # launching from the equatorial on the side of the planet facing away from the sun
    t0 = 0                                  # launching at the beginning of the simulation of planetary orbits

    r0, r, v, sim_launch_duration, v_orbit = rocket_launch_generalized(t0, phi_launch, max_time, dt, thrust_f, initial_m, mass_loss_rate/3)

    mission.set_launch_parameters(thrust = thrust_f, 
                                mass_loss_rate = mass_loss_rate/3, 
                                initial_fuel_mass = fuel_m, 
                                estimated_launch_duration = max_time, 
                                launch_position = r0, 
                                time_of_launch = t0)

    mission.launch_rocket()
    mission.verify_launch_result(r[-1])

    ''' finding the angle that the spacecraft is facing '''

    mission.take_picture(filename='sky_picture.png', full_sky_image_path='himmelkule.npy')
    phi0 = find_phi0('sky_picture.png')

    ''' finding the velocity of the spacecraft '''

    dlambda_sc = np.array(mission.measure_star_doppler_shifts())
    print(f"The reference stars' Doppler shifts at the spacecraft are Δλ1 = {dlambda_sc[0]} nm, Δλ2 = {dlambda_sc[1]} nm")
    vxy = sc_velocity(dlambda_sun, dlambda_sc, phi)

    ''' finding the position of the spacecraft '''

    distances = mission.measure_distances()                     # the spacecraft's distance from each body in the solar system immediately after launch [AU]
    r_s = trilateration(t0, sim_launch_duration/yr, distances)

    ''' verification '''

    mission.verify_manual_orientation(r_s, vxy, phi0)


if __name__ == '__main__':
    main()


'''
RESULTS:

        TESTING THE SOFTWARE:
    The angle that the spacecraft was facing when the given picture was taken was at phi = 0.0 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 20.0 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 43.5 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 91.1 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 140.0 degrees
    The angle that the spacecraft was facing when the given picture was taken was at phi = 190.0 degrees
    The spacecraft's (ϕ1, ϕ2)-velocity relative to our sun is vϕ1 = 0.072 AU/yr, vϕ2 = 0.491 AU/yr
    The spacecraft's (x, y)-velocity relative to our sun is vx = 0.471 AU/yr, vy = 0.141 AU/yr

        LAUNCH:
    The simulated rocket launch took 898 seconds, which is approximately 14 minutes
    The spacecraft's distance from the surface of Doofenshmirtz is 2304.07 km when reaching the escape velocity of 9289.98
    Its total mass was then down to 12115.53 kg, which means it lost a total of 33984.47 kg fuel during the launch
    Its coordinates relative to the launch site are x = 2286.84 km, y = 281.21 km
    Its coordinates relative to the sun are x = 529722889.74 km, y = 24060.17 km
    Its velocity components relative to Doofenshmirtz are vx = 9288.82 m/s, vy = 225.60 m/s
    Its velocity components relative to the sun are vx = 9280.59 m/s, vy = 26705.52 m/s
    Rocket was moved up by 4.50388e-06 m to stand on planet surface.
    New launch parameters set.
    Launch completed, reached escape velocity in 897.19 s.
    Your spacecraft position was satisfyingly calculated. Well done!
    *** Achievement unlocked: No free launch! ***

        USING THE SOFTWARE AFTER LAUNCH:
    Picture written to sky_picture.png.
    The angle that the spacecraft was facing when the given picture was taken was at phi = 161.0 degrees
    The reference stars' Doppler shifts at the spacecraft are Δλ1 = -0.0545409165290725 nm, Δλ2 = -0.02755249268403932 nm
    The spacecraft's (ϕ1, ϕ2)-velocity relative to our sun s is vϕ1 = 5.328 AU/yr, vϕ2 = 3.146 AU/yr
    The spacecraft's (x, y)-velocity relative to our sun is vx = 1.956 AU/yr, vy = 5.664 AU/yr
    Planets' positions relative the sun after launch:
        Blossom: [5.635, 0.978] AU
          Flora: [13.652, 10.805] AU
    Distances after launch:
        Blossom: 2.311 AU
          Flora: 14.798 AU
            sun: 3.541 AU
    The spacecraft's position relative to the sun is x = 3.54098 AU, y = 0.000222488 AU immediately after the launch

        MANUAL ORIENTATION:
    Pointing angle after launch correctly calculated. Well done!
    Velocity after launch correctly calculated. Well done!
    Position after launch correctly calculated. Well done!
    Your manually inferred orientation was satisfyingly calculated. Well done!
    *** Achievement unlocked: Well-oriented! ***
'''
