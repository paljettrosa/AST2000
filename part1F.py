import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from part1E import thrust_f, mass_loss_rate, fuel_m, sim_launch_duration, r0, r, v

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

launch_position = np.array([utils.m_to_AU(r0[0]), utils.m_to_AU(r0[1])])
extra_time = 120

'''
RIKTIG METODE??? SKAL EXTRA_TIME OPPDATERES UTIFRA RESULTAT???
HVORDAN INKLUDERE HJEMPLANETENS ORBITAL VELOCITY???
'''

mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = sim_launch_duration + extra_time, 
                              launch_position = launch_position, 
                              time_of_launch = 0.0)

mission.launch_rocket()

position_after_launch = np.array([utils.m_to_AU(r[-1][0]) + utils.s_to_yr(extra_time)*utils.m_pr_s_to_AU_pr_yr(v[-1][0]), 
                                  utils.m_to_AU(r[-1][1]) + utils.s_to_yr(extra_time)*utils.m_pr_s_to_AU_pr_yr(v[-1][1])])

mission.verify_launch_result(position_after_launch = position_after_launch)

'''
FRA 1E:

The rocket's position is at x = 7.70489e+08 km, y = 313.103 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 9045.28, it's
velocity has a horisontal component of 9039.28 m/s and a vertical
component of 329.583 m/s
The simulated rocket launch took 950 seconds, which is
approximately 15 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 11156.9 kg, which means it lost a total of 89943.1 kg fuel
during the launch


FRA 1F:
    
Rocket was moved up by 9.08328e-05 m to stand on planet surface.
New launch parameters set.
Launch completed, reached escape velocity in 1042.03 s.
Your spacecraft position deviates too much from the correct position.
The deviation is approximately 0.000159793 AU.
Make sure you have included the rotation and orbital velocity of your home planet.
Note that units are AU and relative the the reference system of the star.
'''
