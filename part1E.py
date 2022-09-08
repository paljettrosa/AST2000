import numpy as np
from tqdm import trange 
#from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1D import N_H2, N_b, mean_f, fl_s, fuel_m, sc_m

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)
'''
print(system.aphelion_angles)
print(system.eccentricities)
print(system.has_moons)
print(system.initial_orbital_angles)
print(system.initial_positions)
print(system.initial_velocities)
'''

def rocket_launch(r0, v0, max_time, dt, tot_f, mlr, fuel_m, sc_m):
    v_esc = np.sqrt(2*const.G*system.masses[0]*const.m_sun/(system.radii[0]*10**3))     # the escape velocity for our home planet [m/s]
    sim_launch_duration = 0                                                             # duration of our simulated rocket launch [s]
    rocket_m = sc_m + fuel_m + const.m_H2*N_H2*N_b                                      # initial rocket mass [kg]
    N = max_time/dt                     # number of time steps
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                           # initial position [m]
    v[0] = v0                           # initial velocity [m/s]
    for i in range(int(N) - 1):
        rocket_m -= mlr                                             # updating the rocket's mass during the launch
        fG = gravity(r[i], rocket_m)                                # the gravitational pull from our home planet [N]
        a = np.array([(tot_f + fG[0])/rocket_m, fG[1]/rocket_m])    # the rocket's total acceleration at current time step [m/s**2]
        v[i+1] = v[i] + a*dt                                        # updated velocity
        r[i+1] = r[i] + v[i+1]*dt                                   # updated position
        #print(np.linalg.norm(v[i+1]))
        if tot_f <= np.linalg.norm(fG):               # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        #if np.linalg.norm(v[i+1]) >= v_esc:
        if v[i+1][0] >= v_esc:                        # checking if the rocket has reached the escape velocity
            r = r[:i+1]
            v = v[:i+1]
            sim_launch_duration = i*dt                # updating the duration of our simulated rocket launch
            break                                     
    return r, v, sim_launch_duration, rocket_m
    
def gravity(r, rocket_m):
    theta = r[0]/np.linalg.norm(r)                                                  # angle between our current 
                                                                                    # positional vector and the x-axis
    abs_fG = - const.G*system.masses[0]*const.m_sun*rocket_m/np.linalg.norm(r)**2   # absolute value of the gravitational pull
    fG = np.array([abs_fG*np.cos(theta), abs_fG*np.sin(theta)])                     # vectorized gravitational pull
    return fG

'''
let's assume that we wish to lauch our rocket from the equator, on the side of the
planet facing away from our sun. then our initial position will be as follows
'''
r0 = np.array([utils.AU_to_m(system.initial_positions[0][0]) + system.radii[0]*10**3, 
                 utils.AU_to_m(system.initial_positions[1][0])])
v0 = np.array([utils.AU_pr_yr_to_m_pr_s(system.initial_velocities[0][0]), 
                 utils.AU_pr_yr_to_m_pr_s(system.initial_velocities[1][0])])

dt = 1                              # time step [s]
max_time = 20*60                    # maximum launch time [s]

tot_f = N_b*mean_f                  # the rocket's total thrust force [N]
mlr = N_b*fl_s                      # mass loss rate [kg/s]

r, v, sim_launch_duration, final_m = rocket_launch(r0, v0, max_time, dt, tot_f, mlr, fuel_m, sc_m)

print(r[0], r[-1])
print(v[-1])
print(sim_launch_duration)
print(final_m)

'''
THRUST FORCE ER FOR LAV, MEN LAUNCH TIME BLIR BARE KORTERE JO FLERE PARTIKLER
VI HAR I GASSBOKSENE. MASSEN BLIR OGSÅ NEGATIV ETTER HVERT, HVA KOMMER DET AV?

VANLIG AT KODEN BRUKER 10 MIN???

HVOR MANGE PARTIKLER PER BOKS???

ER DET NOE POENG I AT FUNKSJONEN SKAL TA INN SÅ MANGE PARAMETERE???

IMPORTER HELLER FUNKSJONER, HA EGNE KODER MED BARE FUNKSJONER SÅNN SOM OSKAR HAR

ENDRE VARIABELNAVN TIL LITT MER BESKRIVENDE???
'''

'''
KODE KJØRT MED 10**5 PARTIKLER PER BOKS

There are 3.5134e+31 particles exiting the combustion chamber per second
The combustion chamber exerts a thrust of 4.85646e+08 N
The combustion chamber loses a mass of 117609 kg/s
The rocket uses a total of 26880.9 kg fuel to boost its speed 10000 m/s
r[0] = [7.70485952e+11 0.00000000e+00], r[-1] = [7.70452089e+11 2.72726185e+07]
v[-1] = [-32348.09765206  22746.13719707]
sim_launch_duration = 0 s
final_m = -141002355.11537945 kg

her har gravitasjon vunnet over thrust, så vi har bevegd oss innover i planeten
og aldri nådd terminal velocity. derfor er tiden 0 sekunder. hvorfor er massen
negativ til slutt?
'''

'''
KODE KJØRT MED 1000 PARTIKLER PER BOKS

There are 3.81e+29 particles exiting the combustion chamber per second
The combustion chamber exerts a thrust of 5.3176e+06 N
The combustion chamber loses a mass of 1275.38 kg/s
The rocket uses a total of 26622.3 kg fuel to boost its speed 10000 m/s
r[0] = [7.70485952e+11 0.00000000e+00], r[-1] = [7.70485976e+11 1.59222960e+05]
v[-1] = [ 7896.85130161 22746.13719743]
sim_launch_duration = 7 s
final_m = 896.9800393969545
'''
