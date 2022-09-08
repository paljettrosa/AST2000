'''
er dette del F??? hva går egt den delen ut på
'''

import numpy as np
from tqdm import trange 
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
'''
importer variablene!!!!!!!!!

import gasboxwnozzle
import fuel_consumption
import rocket_launch
import gravity
'''

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = sim_launch_duration #eller bare et tall?, 
                              launch_position = r0, 
                              time_of_launch = 0)

mission.launch_rocket(time =, step =)

mission.verify_launch_result(position_after_launch =)
