import numpy as np
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

'''
A: The Habitable Zone
'''

R = system.radii*1e3
sun_R = system.star_radius*1e3
T = system.star_temperature
init_pos = system.initial_positions
r = np.zeros(len(planets))
for i in range(len(planets)):
    r[i] = utils.AU_to_m(np.linalg.norm(np.array([init_pos[0][i], init_pos[1][i]])))

def habitable_zone(planets, R, sun_R, T, r):
    sigma = 5.670374419*10**(-8)                                    # Stefan-Boltzmann's constant [W/(m**2*K**4)]
    L = sigma*T**4                                                  # the sun's luminosity [W/m**2]
    
    A = 4*np.pi*sun_R**2                                            # the surface area of the sun [m**2]
    
    a = np.zeros(len(planets))
    F = np.zeros(len(planets))
    tot_E = np.zeros(len(planets))
    min_a_panel = np.zeros(len(planets))
    surface_T = np.zeros(len(planets))
    
    lower_boundary = 260                                            # the lower boundary for the habitable zone [K]
    upper_boundary = 390                                            # the upper boundary for the habitable zone [K]
    allowed_deviation = 15                                          # how much the surface temperature can deviate and still be counted as a habitable planet [K]
    
    for i in range(len(planets)):
        a[i] = 4*np.pi*R[i]**2                                      # the surface area of the current planet [m**2]
        print(f"{planets[i][0]}'s surface area is {a[i]*1e-6:g} km^2")
        
        F[i] = A*L/(4*np.pi*r[i]**2)                                # the received flux per square meter for the current planet [W/m**2]
        print(f"The flux received by {planets[i][0]} is {F[i]:.2f} W/m^2")
        
        min_a_panel[i] = 40/(F[i]*0.12)                             # the minimum surface area of the solar panel on the lander needed for the current planet [m**2]
        print(f"The minimum solar panel area needed for the lander unit to function on {planets[i][0]} is {min_a_panel[i]:.2f} m^2")
        
        tot_E[i] = F[i]*2*np.pi*R[i]**2                             # the total energy received per second by the current planet [W]
        print(f"The total energy received by {planets[i][0]} is {tot_E[i]:g} W")
        
        surface_T[i] = (F[i]/sigma)**(1/4)                          # the mean surface temperature for the current planet [K]
        if surface_T[i] < lower_boundary - allowed_deviation:
            print(f"{planets[i][0]}'s surface temperature is {surface_T[i]:.2f} K, so it's too cold to be within the habitable zone\n")
        elif lower_boundary - allowed_deviation <= surface_T[i] < lower_boundary:
            print(f"{planets[i][0]}'s surface temperature is {surface_T[i]:.2f} K, so it's within the habitable zone, but it's rather cold\n")
        elif lower_boundary <= surface_T[i] <= upper_boundary:
            print(f"{planets[i][0]}'s surface temperature is {surface_T[i]:.2f} K, so it's within the habitable zone\n")
        elif upper_boundary < surface_T[i] <= upper_boundary + allowed_deviation:
            print(f"{planets[i][0]}'s surface temperature is {surface_T[i]:.2f} K, so it's within the habitable zone, but it's rather hot\n")
        elif upper_boundary + allowed_deviation < surface_T[i]:
            print(f"{planets[i][0]}'s surface temperature is {surface_T[i]:.2f} K, so it's too hot to be within the habitable zone\n")

habitable_zone(planets, R, sun_R, T, r)

'''
Doofenshmirtz's surface area is 4.99734e+08 km^2
The flux received by Doofenshmirtz is 3052.52 W/m^2
The minimum solar panel area needed for the lander unit to function on Doofenshmirtz is 0.11 m^2
The total energy received by Doofenshmirtz is 7.62725e+17 W
Doofenshmirtz's surface temperature is 481.68 K, so it's too hot to be within the habitable zone

Blossom's surface area is 1.17768e+08 km^2
The flux received by Blossom is 1170.18 W/m^2
The minimum solar panel area needed for the lander unit to function on Blossom is 0.28 m^2
The total energy received by Blossom is 6.89053e+16 W
Blossom's surface temperature is 379.02 K, so it's within the habitable zone

Bubbles's surface area is 1.04671e+11 km^2
The flux received by Bubbles is 44.88 W/m^2
The minimum solar panel area needed for the lander unit to function on Bubbles is 7.43 m^2
The total energy received by Bubbles is 2.34904e+18 W
Bubbles's surface temperature is 167.73 K, so it's too cold to be within the habitable zone

Buttercup's surface area is 5.74932e+08 km^2
The flux received by Buttercup is 287.88 W/m^2
The minimum solar panel area needed for the lander unit to function on Buttercup is 1.16 m^2
The total energy received by Buttercup is 8.27549e+16 W
Buttercup's surface temperature is 266.93 K, so it's within the habitable zone

Flora's surface area is 7.14339e+10 km^2
The flux received by Flora is 126.26 W/m^2
The minimum solar panel area needed for the lander unit to function on Flora is 2.64 m^2
The total energy received by Flora is 4.5096e+18 W
Flora's surface temperature is 217.23 K, so it's too cold to be within the habitable zone

Stella's surface area is 3.50798e+07 km^2
The flux received by Stella is 702.41 W/m^2
The minimum solar panel area needed for the lander unit to function on Stella is 0.47 m^2
The total energy received by Stella is 1.23202e+16 W
Stella's surface temperature is 333.61 K, so it's within the habitable zone

Aisha's surface area is 1.00318e+10 km^2
The flux received by Aisha is 71.58 W/m^2
The minimum solar panel area needed for the lander unit to function on Aisha is 4.66 m^2
The total energy received by Aisha is 3.5902e+17 W
Aisha's surface temperature is 188.49 K, so it's too cold to be within the habitable zone
'''