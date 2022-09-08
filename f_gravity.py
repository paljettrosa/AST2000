import numpy as np
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

def gravity(r, rocket_m):
    theta = r[0]/np.linalg.norm(r)                                                  # angle between our current 
                                                                                    # positional vector and the x-axis
    abs_fG = - const.G*system.masses[0]*const.m_sun*rocket_m/np.linalg.norm(r)**2   # absolute value of the gravitational pull
    fG = np.array([abs_fG*np.cos(theta), abs_fG*np.sin(theta)])                     # vectorized gravitational pull
    return fG