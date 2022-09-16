import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from f_rocket_launch import rocket_launch, x0, y0
from part1E import max_time, dt, R, v_esc, initial_m, thrust_f, fuel_m, mass_loss_rate, r, sim_launch_duration

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

r0_sun = np.array([x0 + R, y0])                       # our rocket's initial position relative to our sun [m]

T = system.rotational_periods[0]*24*60*60             # our planet's rotational period [s]
omega = 2*np.pi/T                                     # our planet's rotational velocity [s**(-1)]
v_rot = - R*omega                                     # our rocket's initial velocity caused by our planet's rotation
                                                      # assuming that our planet's rotational velocity is positive [m/s]
v_orbit = r[-1][1]/sim_launch_duration                # our rocket's initial velocity caused by our planet's orbital
                                                      # velocity, approximated by our simulation [m/s]
v0_sun = np.array([0.0, v_orbit + v_rot])             # our rocket's initial velocity relative to our sun [m/s]

thrust_f = 3*thrust_f
initial_m = initial_m + fuel_m
fuel_m = 2*fuel_m

r, v, sim_launch_duration, final_m = rocket_launch(r0_sun, v0_sun, v_esc, max_time, dt, thrust_f, initial_m, mass_loss_rate)

print(f"The rocket's position is at x = {r[-1][0]/10**3:g} km, y = {r[-1][1]/10**3:g} km\nwhen it reaches the escape velocity")
print(f"When the rocket reaches it's escape velocity of {np.linalg.norm(v[-1]):g}, it's\nvelocity has a horisontal component of {v[-1][0]:g} m/s and a vertical\ncomponent of {v[-1][1]:g} m/s")
print(f"The simulated rocket launch took {sim_launch_duration} seconds, which is\napproximately {int(sim_launch_duration/60)} minutes")
print(f"When the rocket reached it's escape velocity, it's total mass was\ndown to {final_m:g} kg, which means it lost a total of {initial_m - final_m:g} kg fuel\nduring the launch")

mission.set_launch_parameters(thrust = thrust_f, 
                              mass_loss_rate = mass_loss_rate, 
                              initial_fuel_mass = fuel_m, 
                              estimated_launch_duration = 1000, 
                              launch_position = utils.m_to_AU(r0_sun), 
                              time_of_launch = 0.0)

mission.launch_rocket()

'''
position_after_launch = np.array([utils.m_to_AU(r[-1][0]), utils.m_to_AU(r[-1][1])])

SKAFF SHORTCUT ELLER SPØR GRUPPELÆRER OM FEIL I KODE

mission.verify_launch_result(position_after_launch)

HUSK Å FORKLARE HVORFOR VI ENDRET THRUST_FORCE OG FUEL_M, OG HVORDAN VI BEREGNET V_ORBIT
'''

'''
FRA SIMULASJON:

The rocket's position is at x = 7.70487e+08 km, y = -1319.74 km
when it reaches the escape velocity
When the rocket reaches it's escape velocity of 9060.74, it's
velocity has a horisontal component of 8071.36 m/s and a vertical
component of -4117.06 m/s
The simulated rocket launch took 445 seconds, which is
approximately 7 minutes
When the rocket reached it's escape velocity, it's total mass was
down to 14109.4 kg, which means it lost a total of 16990.6 kg fuel
during the launch


FRA OPPSKYTNING:
    
Rocket was moved up by 9.08328e-05 m to stand on planet surface.
New launch parameters set.
Launch completed, reached escape velocity in 491.97 s.
Your spacecraft position deviates too much from the correct position.
The deviation is approximately 8.47092e-05 AU.
Make sure you have included the rotation and orbital velocity of your home planet.
Note that units are AU and relative the the reference system of the star.
'''
