import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from f_gasboxwnozzle import gasboxwnozzle
from f_fuel_consumption import fuel_consumption
from part1C import N_H2, m_H2, my, sigma, L, time, steps

utils.check_for_newer_version()
seed = utils.get_seed('hask')
system = SolarSystem(seed)
mission = SpaceMission(seed)

r, v, exiting, f = gasboxwnozzle(my, sigma, N_H2, m_H2, L, time, steps)

particles_s = exiting/time              # the number of particles exiting per second [s**(-1)]
mean_f = f/steps                        # the box force averaged over all time steps [N]
fuel_loss_s = particles_s*m_H2          # the total fuel loss per second [kg/s]

A_box = L*L                             # area of one gasbox [m**2]
A_spacecraft = mission.spacecraft_area  # area of our spacecraft's cross section [m**2]
N_box = (A_spacecraft/A_box)*50         # number of gasboxes                   
thrust_f = N_box*mean_f                 # the combustion chamber's total thrust force [N]

print(f'There are {particles_s*N_box:g} particles exiting the combustion chamber per second')
print(f'The combustion chamber exerts a thrust of {thrust_f:g} N')
print(f'The combustion chamber loses a mass of {fuel_loss_s*N_box:g} kg/s')

spacecraft_m = mission.spacecraft_mass                  # mass of rocket without fuel [kg]
fuel_m = 10**5                                          # mass of feul [kg]
initial_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box     # initial rocket mass [kg]

delta_v = 10**4                                         # change in the rocket's velocity [m/s]

tot_fuel_loss = fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, delta_v)

print(f'The rocket uses a total of {tot_fuel_loss:g} kg fuel to boost its speed {delta_v:g} m/s')

'''
There are 2.81344e+28 particles exiting the combustion chamber per second
The combustion chamber exerts a thrust of 389005 N
The combustion chamber loses a mass of 94.1784 kg/s
The rocket uses a total of 244764 kg fuel to boost its speed 10000 m/s
'''
