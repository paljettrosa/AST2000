import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from f_rocket_launch import rocket_launch
from part1D import N_H2, m_H2, N_box, mean_f, fuel_loss_s, fuel_m, spacecraft_m

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

R = system.radii[0]*10**3                               # our planet's radius [m]
M = system.masses[0]*const.m_sun                        # our planet's mass [kg]

'''
let's assume that we wish to lauch our rocket from the equator, on the side of the
planet facing away from our sun. then our initial position will be as follows
'''
r0 = np.array([R, 0.0])

'''
when our rocket is moving away from our home planet, it's still within it's gravitational
field, and therefore moves around our sun with the same velocity as our planet does. because
of this, we ignore this contribution to our rocket's initial velocity. the contribution
coming from our planet's rotational velocity is still important to include, as our rocket
stops rotating almost immediately after leaving our planet's surface. our rocket will then
have a vertical velocity component relative to our planet, assuming that our planet rotates
around the x-axis when it's in it's initial position
'''

v0 = np.array([0.0, 0.0])

v_esc = np.sqrt(2*const.G*M/R)                          # the escape velocity for our home planet [m/s]

dt = 1                                                  # time step [s]
max_time = 20*60                                        # maximum launch time [s]

initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]
thrust_f = N_box*mean_f                                 # the rocket's total thrust force [N]
mass_loss_rate = N_box*fuel_loss_s                      # mass loss rate [kg/s]

r, v, sim_launch_duration, final_m = rocket_launch(r0, v0, v_esc, max_time, dt, thrust_f, initial_m, mass_loss_rate)

print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][1]/10**3:g} km\nwhen it reaches the escape velocity")
print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g}, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {final_m:g} kg, which means it lost a total of {initial_m - final_m:g} kg fuel\nduring the launch")

'''
The rocket's position is at x = 6080.32 km, y = -480.087 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 9004.12, it's
velocity has a horisontal component of 8685.77 m/s and a vertical
component of -2373.1 m/s
The simulated rocket launch took 385 seconds, which is
approximately 6 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 1299.6 kg, which means it lost a total of 14800.4 kg fuel
during the launch
'''
