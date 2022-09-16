import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from f_gravity import gravity

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

x0 = utils.AU_to_m(system.initial_positions[0][0])                      # our planet's initial x-coordinate relative to our sun [m]
y0 = utils.AU_to_m(system.initial_positions[1][0])                      # our planet's initial y-coordinate relative to our sun [m]

def rocket_launch(r0, v0, v_esc, max_time, dt, thrust_f, initial_m, mass_loss_rate):
    sim_launch_duration = 0             # duration of our simulated rocket launch [s]
    rocket_m = initial_m                # the rocket's total mass [kg]
    N = max_time/dt                     # number of time steps
    r = np.zeros((int(N), 2))   
    v = np.zeros((int(N), 2))
    r[0] = r0                           # initial position [m]
    v[0] = v0                           # initial velocity [m/s]
    for i in range(int(N) - 1):
        '''
        checking what referance system we're using
        '''
        if r[i][0] - x0 < 0:
            fG = gravity(r[i], rocket_m)                                # the gravitational pull from our home planet [N]
        elif r[i][0] - x0 >= 0:
            fG = gravity(r[i] - np.array([x0, y0]), rocket_m)           # the gravitational pull from our home planet [N]
        a = np.array([(thrust_f + fG[0])/rocket_m, fG[1]/rocket_m])     # the rocket's total acceleration at current time step [m/s**2]
        v[i+1] = v[i] + a*dt                                            # updated velocity
        r[i+1] = r[i] + v[i+1]*dt                                       # updated position
        rocket_m -= mass_loss_rate*dt                                   # updating the rocket's mass during the launch
        if thrust_f <= np.linalg.norm(fG):                              # checking if the thrust force is too low       
            print('Thrust force is too low!')
            break
        if np.linalg.norm(v[i+1]) >= v_esc:           # checking if the rocket has reached the escape velocity
            r = r[:i+1]
            v = v[:i+1]
            sim_launch_duration = i*dt                # updating the duration of our simulated rocket launch
            break                                     
    return r, v, sim_launch_duration, rocket_m
