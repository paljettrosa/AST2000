import ast2000tools.constants as con
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
import ast2000tools.utils as utils
import numpy as np


seed = utils.get_seed("somiamc")
code_launch_results = 10978  # insert code here
# print(seed)

mission = SpaceMission(seed)
system = SolarSystem(seed)

shortcut = SpaceMissionShortcuts(mission, [10978])

"""
Documentation
get_launch_results():

------------------------------------------------------------------------
get_launch_results() returns the results from the previous launch, i.e.,
the results from previous mission.launch_rocket()

Returns
-------
fuel_consumed  :  float
    The total mass of fuel burned during the launch, in KILOGRAMS.

time_after_launch  :  float
    The time when the launch was completed, in YEARS since the initial
    solar system time.

pos_after_launch  :  ndarray
    Array of shape (2,) containing the x and y-position of the space-
    craft, in ASTRONOMICAL UNITS relative to the star.

vel_after_launch  :  ndarray
    Array of shape (2,) containing the x and y-velocity of the space-
    craft, in ASTRONOMICAL UNITS PER YEAR relative to the star.

Raises
------
RuntimeError
    When none of the provided codes are valid for unlocking this method.
RuntimeError
    When called before mission.launch_rocket() has been called success-
    fully.
------------------------------------------------------------------------

"""

thrust = # insert the thrust force of your spacecraft here
mass_loss_rate = # insert the mass loss rate of your spacecraft here
initial_fuel_mass = # insert your initial fuel mass here
launch_time = # insert your simulated launch time IN SECONDS here

radius = system.radii[0] * 1e3                      # m
R0 = system.initial_positions[:, 0]                 # AU
R0 = R0 + np.array([radius, 0]) / con.AU            # AU
t0 = 0                                              # yr

mission.set_launch_parameters(thrust, mass_loss_rate, initial_fuel_mass,
    launch_time, R0, t0)
mission.launch_rocket()

(
    fuel_consumed,
    time_after_launch,
    pos_after_launch,
    vel_after_launch,
) = shortcut.get_launch_results()

mission.verify_launch_results(pos_after_launch)
