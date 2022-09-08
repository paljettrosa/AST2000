import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from f_rocket_launch import rocket_launch
from part1D import m_H2, N_H2, N_box, mean_f, fuel_loss_s, fuel_m, spacecraft_m

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

'''
let's assume that we wish to lauch our rocket from the equator, on the side of the
planet facing away from our sun. then our initial position will be as follows
'''
r0 = np.array([utils.AU_to_m(system.initial_positions[0][0]) + system.radii[0]*10**3, 
                 utils.AU_to_m(system.initial_positions[1][0])])
v0 = np.array([utils.AU_pr_yr_to_m_pr_s(system.initial_velocities[0][0]), 
                 utils.AU_pr_yr_to_m_pr_s(system.initial_velocities[1][0])])

dt = 1                                                  # time step [s]
max_time = 20*60                                        # maximum launch time [s]

initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
thrust_f = N_box*mean_f                                 # the rocket's total thrust force [N]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

r, v, sim_launch_duration, final_m = rocket_launch(r0, v0, max_time, dt, initial_m, thrust_f, mass_loss_rate)

'''
print(r[0], r[-1])
print(v[-1])
print(sim_launch_duration)
print(final_m)
'''

print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][0]/10**3:g} km\nwhen it reaches the escape velocity")
print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g}, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {final_m:g} kg, which means it lost a total of {initial_m - final_m:g} kg fuel\nduring the launch")

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
