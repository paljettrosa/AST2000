#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1BC import gasboxwnozzle, m_H2, my, sigma, L, time, steps

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)
plt.rcParams.update({'font.size': 14})

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = const.G                                          # gravitation constant [m**3s**(-2)kg**(-1)]
AU = const.AU                                        # one astronomical unit [m]
yr = const.yr                                        # one year [s]

M = system.masses[0]*const.m_sun                     # Doofenshmirtz' mass [kg]
R = system.radii[0]*1e3                              # Doofenshmirtz' radius [m]
T = system.rotational_periods[0]*const.day           # Doofenshmirtz' rotational period [s]

spacecraft_m = spacecraft_m = mission.spacecraft_mass                  # mass of rocket without fuel [kg]
spacecraft_A = mission.spacecraft_area                                 # area of our spacecraft's cross section [m**2]

f = np.load(r'/Users/paljettrosa/Documents/AST2000/planet_trajectories.npz')
times = f['times']
dt_p = times[1]*yr
r_p = np.einsum("ijk->kji", f['planet_positions'])*AU


'''
C: Generalized Launch Codes
'''

def rocket_launch_generalized(t0, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate):    

    '''
    generalized rocket launch function
    '''

    ''' we use SI-units during the calculations to avoid as many round-off errors as possible '''    

    N = max_time/dt                                                     # number of time steps
    N_p = int(t0*yr/dt_p)                                               # finding out at which time step we start the rocket launch from

    r0_p = r_p[N_p, 0]                                                  # Doofenshmirtz's initial positon relative to our sun [m] 
    r0 = np.array([np.cos(phi)*R + r0_p[0], np.sin(phi)*R + r0_p[1]])
    
    omega = 2*np.pi/T                                                   # Doofenshmirtz's rotational velocity [s**(-1)]
    v_rot = R*omega                                                     # our rocket's initial velocity caused by the planet's rotation [m/s]
    
    v_orbitx = (r_p[N_p+1, 0, 0] - r_p[N_p, 0, 0])/dt_p                 # our rocket's approximated horizontal velocity caused by the planet's orbital velocity [m/s]
    v_orbity = (r_p[N_p+1, 0, 1] - r_p[N_p, 0, 1])/dt_p                 # our rocket's approximated vertical velocity caused by the planet's orbital velocity [m/s]
    
    vx0 = v_orbitx - np.sin(phi)*v_rot
    vy0 = v_orbity + np.cos(phi)*v_rot
    v0 = np.array([vx0, vy0])                                           # our rocket's initial velocity relative to our sun [m/s]         
   
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                                                           # initial position [m]
    v[0] = v0                                                           # initial velocity [m/s]

    sim_launch_duration = 0                                             # duration of our simulated rocket launch [s]
    rocket_m = initial_m                                                # the rocket's total mass [kg]
    
    for i in range(int(N) - 1):

        pos = r[i] - (r0_p + np.array([v_orbitx, v_orbity])*i*dt)
        fG = - G*M*rocket_m/np.linalg.norm(pos)**3*pos                  # the gravitational pull from Doofenshmirtz [N]
        a = np.array([(np.cos(phi)*thrust_f + fG[0])/rocket_m, (np.sin(phi)*thrust_f + fG[1])/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        
        v[i+1] = v[i] + a*dt/2                                            
        r[i+1] = r[i] + v[i+1]*dt   
        
        pos = r[i+1] - (r0_p + np.array([v_orbitx, v_orbity])*(i+1)*dt)
        fG = - G*M*rocket_m/np.linalg.norm(pos)**3*pos                  # the gravitational pull from Doofenshmirtz [N]
        a = np.array([(np.cos(phi)*thrust_f + fG[0])/rocket_m, (np.sin(phi)*thrust_f + fG[1])/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]

        v[i+1] = v[i+1] + a*dt/2  
        
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        if rocket_m <= spacecraft_m:                                    # checking if we run out of fuel
            print('Ran out of fuel!')
            break
        
        planet_sun = np.array([r0_p[0] + v_orbitx*(i+1)*dt, r0_p[1] + v_orbity*(i+1)*dt])   # Doofenshmirtz' curretn position relativee to our sun [m] 
        v_esc = np.sqrt(2*G*M/np.linalg.norm(r[i+1] - planet_sun))      # the current escape velocity [m/s]

        if np.linalg.norm(v[i+1] - v0) >= v_esc:                        # checking if the rocket has reached the escape velocity
            r = r[:i+2]
            v = v[:i+2]
            sim_launch_duration = (i+1)*dt                              # the duration of our simulated rocket launch

            x_p = r0_p[0] + v_orbitx*sim_launch_duration                # Doofenshmirtz's updated x-position
            y_p = r0_p[1] + v_orbity*sim_launch_duration                # Doofenshmirtz's updated y-position
            distance = np.linalg.norm(r[-1] - np.array([x_p, y_p]) - r0 + r0_p)
            
            print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is approximately {int(sim_launch_duration/60)} minutes")
            print(f"The spacecraft's distance from the surface of Doofenshmirtz is {distance*1e-3:.2f} km when reaching the escape velocity of {np.linalg.norm(v[-1] - v0):.2f}")
            print(f"Its total mass was then down to {rocket_m:.2f} kg, which means it lost a total of {initial_m - rocket_m:.2f} kg fuel during the launch")
            print(f"Its coordinates relative to the launch site are x = {(r[-1, 0] - x_p - r0[0] + r0_p[0])*1e-3:.2f} km, y = {(r[-1, 1] - y_p - r0[1] + r0_p[1])*1e-3:.2f} km")
            print(f"Its coordinates relative to the sun are x = {r[-1][0]*1e-3:.2f} km, y = {r[-1][1]*1e-3:.2f} km")
            print(f"Its velocity components relative to Doofenshmirtz are vx = {v[-1][0] - v_orbitx:.2f} m/s, vy = {v[-1][1] - v_orbity:.2f} m/s")
            print(f"Its velocity components relative to the sun are vx = {v[-1][0]:.2f} m/s, vy = {v[-1][1]:.2f} m/s")
            break 
    
    ''' changing to astronomical units'''
    
    r0 = utils.m_to_AU(r0)
    r = utils.m_to_AU(r)
    v = utils.m_pr_s_to_AU_pr_yr(v)
    v_orbit = utils.m_pr_s_to_AU_pr_yr(np.array([v_orbitx, v_orbity]))
    
    return r0, r, v, sim_launch_duration, v_orbit



def main():

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
    t0 = 0                                  # launching at the beginning of the simulation of planetary orbits

    r0, r, v, sim_launch_duration, v_orbit = rocket_launch_generalized(t0, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate)

    mission.set_launch_parameters(thrust = thrust_f, 
                                mass_loss_rate = mass_loss_rate, 
                                initial_fuel_mass = fuel_m, 
                                estimated_launch_duration = 1000, 
                                launch_position = r0, 
                                time_of_launch = t0)

    mission.launch_rocket()
    mission.verify_launch_result(r[-1])
    print('\n')


    ''' launching with engine adjustments '''

    r0, r, v, sim_launch_duration, v_orbit = rocket_launch_generalized(t0, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate/3)

    N_start = int(t0*yr/dt_p)
    N_stop = int((t0*yr + sim_launch_duration)/dt_p)
    trajectory_D = r_p[N_start:N_stop+1, 0]

    plt.plot((r[:, 0]*AU - trajectory_D[:, 0])*1e-3, (r[:, 1]*AU - trajectory_D[:, 1])*1e-3, color = 'mediumvioletred')
    plt.xlabel('x-position [km]')
    plt.ylabel('y-position [km]')
    plt.title("The rocket's trajectory throughout the\nsimulated launch sequence")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('/Users/paljettrosa/Documents/AST2000/trajectory_launch_generalized1.pdf')

    t = np.linspace(0, sim_launch_duration, len(r))
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(np.linalg.norm(v - v_orbit, axis=1))*1e-3, color = 'cornflowerblue')
    plt.xlabel('time [s]')
    plt.ylabel(r'absolute velocity ($v$) [$\frac{km}{s}$]')
    plt.title("The rocket's absolute velocity throughout\nthe simulated launch sequence")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('/Users/paljettrosa/Documents/AST2000/velocity_launch_generalized1.pdf')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)
    ax1.plot(t, utils.AU_pr_yr_to_m_pr_s(v[:, 0] - v_orbit[0])*1e-3, color = 'mediumpurple')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel(r'velocity ($v_x$) [$\frac{km}{s}$]')
    ax2.plot(t, utils.AU_pr_yr_to_m_pr_s(v[:, 1] - v_orbit[1])*1e-3, color = 'violet')
    ax2.set_ylabel(r'velocity ($v_y$) [$\frac{km}{s}$]')
    fig.suptitle("The rocket's velocity components throughout the\nsimulated launch sequence")
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/xy_velocity_launch_generalized1.pdf')

    mission.set_launch_parameters(thrust = thrust_f, 
                                mass_loss_rate = mass_loss_rate/3, 
                                initial_fuel_mass = fuel_m, 
                                estimated_launch_duration = 1000, 
                                launch_position = r0, 
                                time_of_launch = t0)

    mission.launch_rocket()
    mission.verify_launch_result(r[-1])
    print('\n')




    ''' 'trying with other parameters '''

    phi = np.pi/2                                           # launching from ϕ = π/2
    t0 = 3                                                  # launching three years after we started simulating the orbits

    r0, r, v, sim_launch_duration, v_orbit = rocket_launch_generalized(t0, phi, max_time, dt, thrust_f, initial_m, mass_loss_rate/3)

    N_start = int(t0*yr/dt_p)
    N_stop = int((t0*yr + sim_launch_duration)/dt_p)
    trajectory_D = r_p[N_start:N_stop+1, 0]

    plt.plot((r[:, 0]*AU - trajectory_D[:, 0])*1e-3, (r[:, 1]*AU - trajectory_D[:, 1])*1e-3, color = 'olivedrab')
    plt.xlabel('x-position [km]')
    plt.ylabel('y-position [km]')
    plt.title("The rocket's trajectory throughout the\nsimulated launch sequence")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('/Users/paljettrosa/Documents/AST2000/trajectory_launch_generalized2.pdf')

    t = np.linspace(0, sim_launch_duration, len(r))
    plt.plot(t, utils.AU_pr_yr_to_m_pr_s(np.linalg.norm(v - v_orbit, axis=1))*1e-3, color = 'plum')
    plt.xlabel('time [s]')
    plt.ylabel(r'absolute velocity ($v$) [$\frac{km}{s}$]')
    plt.title("The rocket's absolute velocity throughout\nthe simulated launch sequence")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig('/Users/paljettrosa/Documents/AST2000/velocity_launch_generalized2.pdf')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)
    ax1.plot(t, utils.AU_pr_yr_to_m_pr_s(v[:, 0] - v_orbit[0])*1e-3, color = 'mediumseagreen')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel(r'velocity ($v_x$) [$\frac{km}{s}$]')
    ax2.plot(t, utils.AU_pr_yr_to_m_pr_s(v[:, 1] - v_orbit[1])*1e-3, color = 'skyblue')
    ax2.set_ylabel(r'velocity ($v_y$) [$\frac{km}{s}$]')
    fig.suptitle("The rocket's velocity components throughout\nthe simulated launch sequence")
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/xy_velocity_launch_generalized2.pdf')

    mission.set_launch_parameters(thrust = thrust_f, 
                                mass_loss_rate = mass_loss_rate/3, 
                                initial_fuel_mass = fuel_m, 
                                estimated_launch_duration = 1000, 
                                launch_position = r0, 
                                time_of_launch = t0)

    mission.launch_rocket()
    mission.verify_launch_result(r[-1])

if __name__ == '__main__':
    main()


'''
RESULTS

    LAUNCHING FROM ϕ = 0 AT YEAR 0:

        SIMULATION RESULTS:
    The simulated rocket launch took 392 seconds, which is approximately 6 minutes
    The spacecraft's distance from the surface of Doofenshmirtz is 737.28 km when reaching the escape velocity of 10520.31
    Its total mass was then down to 1520.79 kg, which means it lost a total of 44579.21 kg fuel during the launch
    Its coordinates relative to the launch site are x = 723.71 km, y = 140.81 km
    Its coordinates relative to the sun are x = 529721330.77 km, y = 10520.94 km
    Its velocity components relative to Doofenshmirtz are vx = 10461.04 m/s, vy = 335.13 m/s
    Its velocity components relative to the sun are vx = 10452.81 m/s, vy = 26815.04 m/s

        LAUNCH RESULTS: 
    Rocket was moved up by 4.50388e-06 m to stand on planet surface.
    New launch parameters set.
    Launch completed, reached escape velocity in 390.78 s.
    Your spacecraft position was satisfyingly calculated. Well done!
    *** Achievement unlocked: No free launch! ***

    
    SAME CONDITIONS, NEW MASS LOSS RATE:

        SIMULATION RESULTS:
    The simulated rocket launch took 896 seconds, which is
    approximately 14 minutes
    The spacecraft's distance from the surface of Doofenshmirtz is 2307.01 km when reaching the escape velocity of 9306.18
    Its total mass was then down to 12134.88 kg, which means it lost a total of 33965.12 kg fuel during the launch
    Its coordinates relative to the launch site are x = 2289.85 km, y = 280.84 km
    Its coordinates relative to the sun are x = 529722892.76 km, y = 24006.84 km
    Its velocity components relative to Doofenshmirtz are vx = 9300.89 m/s, vy = 226.33 m/s
    Its velocity components relative to the sun are vx = 9292.66 m/s, vy = 26706.24 m/s

        LAUNCH RESULTS:
    Rocket was moved up by 4.50388e-06 m to stand on planet surface.
    New launch parameters set.
    Note: Existing launch results were cleared.
    Launch completed, reached escape velocity in 894.83 s.
    Your spacecraft position was satisfyingly calculated. Well done!
    *** Achievement unlocked: No free launch! ***


    LAUNCHING FROM ϕ = π/2 THREE YEARS AFTER THE
    PLANETARY ORBIT SIMULATION STARTED:

        SIMULATION RESULTS:
    The simulated rocket launch took 896 seconds, which is approximately 14 minutes
    The spacecraft's distance from the surface of Doofenshmirtz is 2306.96 km when reaching the escape velocity of 9305.83
    Its total mass was then down to 12120.12 kg, which means it lost a total of 33979.88 kg fuel during the launch
    Its coordinates relative to the launch site are x = -280.84 km, y = 2289.81 km
    Its coordinates relative to the sun are x = 76844539.99 km, y = -519043494.03 km
    Its velocity components relative to Doofenshmirtz are vx = -226.32 m/s, vy = 9304.68 m/s
    Its velocity components relative to the sun are vx = 26259.74 m/s, vy = 12937.24 m/s

        LAUNCH RESULTS:
    ValueError: Launch position deviation from home planet surface is 2.77071e+08 m but must not exceed 63061.6 m.
'''