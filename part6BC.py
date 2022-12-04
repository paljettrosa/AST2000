#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts 

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)
shortcut = SpaceMissionShortcuts(mission, [10978]) 
plt.rcParams.update({'font.size': 12})

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = const.G                         # gravitation constant [m^3*s^(-2)*kg^(-1)]
c = const.c                         # speed of light [m/s]
m_sun = const.m_sun                 # solar mass [kg]

gamma = 1.4                         # the heat capacity ratio
k_B = const.k_B                     # Boltzmann constant [m^2*kg/s^2/K]
u = 1.661e-27                       # atomic unit of mass [kg]
mH = 1.00784*1.661e-27              # mass of a hydrogen atom [kg]

m_O2 = 2*15.999*u                   # mass of 02 molecule [kg]
m_H2O = (2*1.00784 + 15.999)*u      # mass of H2O molecule [kg]
m_CO2 = (12.0107 + 2*15.99)*u       # mass of CO2 molecule [kg]
m_CH4 = (12.0107 + 4*1.00784)*u     # mass of CH4 molecule [kg]
m_CO = (12.0107 + 15.999)*u         # mass of CO molecule [kg]
m_N2O = (2*14.0067 + 15.999)*u      # mass of N2O molecule [kg]


'''
B. Spectral Analysis of the Atmosphere
'''

'''
Downloading the Data
'''

spectrum_data = []
with open('spectrum_seed56_600nm_3000nm.txt', 'r') as infile:
    for line in infile:
        spectrum_data.append([float(line.split()[0]), float(line.split()[1])])
    infile.close()
spectrum_data = np.array(spectrum_data)

noise_data = []
with open('sigma_noise.txt', 'r') as infile:
    for line in infile:
        noise_data.append([float(line.split()[0]), float(line.split()[1])])
    infile.close()
noise_data = np.array(noise_data)

data = np.zeros((np.shape(noise_data)[0], 3))
data[:, 0] = spectrum_data[:, 0]
data[:, 1] = spectrum_data[:, 1]
data[:, 2] = noise_data[:, 1]


'''
The Statistical Analysis
'''

def chisquare_method(gas, m, lambda0, color1, color2):
    vmax = 1e4                                                  # upper boundary for the spacecraft's velocity relative to the planet
    Tmax = 450                                                  # upper boundary for the atmospheric temperature
    dlambda = (vmax + np.sqrt(2*k_B*Tmax/m))*lambda0/c          # maximum possible Doppler shift that the spacecraft can observe for this element's spectral line at this wavelength

    ''' 
    slicing wavelength data to only look at intervals where 
    there realistically could be a Doppler shift 
    '''

    tol = 1e-3
    lower_idx = np.where(abs(data[:, 0] - np.round(lambda0 - dlambda, 3)) < tol)[0][0]      
    upper_idx = np.where(abs(data[:, 0] - np.round(lambda0 + dlambda, 3)) < tol)[0][-1]    
    sliced_data = data[lower_idx:upper_idx]                                 
    sliced_data[:, 0] = sliced_data[:, 0]*1e-9                  # changing the wavelength units to meters

    Fmin = np.linspace(0.7, 1, 50)
    lambdas = np.linspace(sliced_data[0, 0], sliced_data[-1, 0], 50)
    T = np.linspace(150, 450, 50)

    def sigma(Ti):

        ''' standard deviation formula '''

        return sliced_data[:, 0]/c*np.sqrt(k_B*Ti/m)
    
    def gaussian_lineprofile(Fmini, lambdai, Ti):

        ''' gaussian line profile formula for modelling spectral lines '''

        return 1 + (Fmini - 1)*np.exp(-1/2*((lambdai - sliced_data[:, 0])/sigma(Ti))**2)

    chisquare = 1e4    
    for i in range(len(Fmin)):
        for j in range(len(lambdas)):
            for k in range(len(T)):
                new_chisquare = np.sum(((sliced_data[:, 1] - gaussian_lineprofile(Fmin[i], lambdas[j], T[k]))/sliced_data[:, 2])**2)
                if Fmin[i] != 0.7 and Fmin[i] != 1:
                    if new_chisquare < chisquare:
                        chisquare = new_chisquare
                        best_Fmin = Fmin[i]
                        best_lambda = lambdas[j]*1e9
                        best_T = T[k]

    sliced_data[:, 0] = sliced_data[:, 0]*1e9                    # changing the wavelength units back to nanometers

    plt.plot(sliced_data[:, 0] - lambda0, sliced_data[:, 1], color = color1, label = 'measured data')
    plt.plot(sliced_data[:, 0] - lambda0, gaussian_lineprofile(best_Fmin, best_lambda, best_T), color = color2, label = 'gaussian line profile')
    plt.legend()
    plt.xlabel("Δλ [nm]")
    plt.ylabel("relative flux")
    plt.title(r"The $\chi^2$" + f"-approximation for {gas}'s" + '\n' + fr"spectral line at $\lambda_0$ = {lambda0} nm")
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'chisquare_{gas}_{lambda0}.pdf')

    return np.array([best_Fmin, best_lambda, best_T])


gases = np.array([['O2', m_O2, 632, 'plum', 'purple'],
                  ['O2', m_O2, 690, 'plum', 'purple'],
                  ['O2', m_O2, 760, 'plum', 'purple'],
                  ['H2O', m_H2O, 720, 'skyblue', 'royalblue'],
                  ['H2O', m_H2O, 820, 'skyblue', 'royalblue'],
                  ['H2O', m_H2O, 940, 'skyblue', 'royalblue'],
                  ['CO2', m_CO2, 1400, 'navajowhite', 'darkorange'],
                  ['CO2', m_CO2, 1600, 'navajowhite', 'darkorange'],
                  ['CH4', m_CH4, 1660, 'khaki', 'darkolivegreen'],
                  ['CH4', m_CH4, 2200, 'khaki', 'darkolivegreen'],
                  ['CO', m_CO, 2340, 'silver', 'black'],
                  ['N2O', m_N2O, 2870, 'pink', 'mediumvioletred']])

atmosphere_table = np.zeros((12, 3))

with open(r'atmosphere_table.txt', 'w') as outfile:
    outfile.write(' Compound & Wavelength |   Flux   |    Doppler Shift    |  Temperature  ' + '\n')
    outfile.write('_______________________|__________|_____________________|_______________' + '\n')
    for i in range(12):
        atmosphere_table[i] = chisquare_method(str(gases[i, 0]), float(gases[i, 1]), int(gases[i, 2]), str(gases[i, 3]), str(gases[i, 4]))
        outfile.write(f'      {gases[i, 0]:>3} - {str(gases[i, 2]):4}       |   {atmosphere_table[i, 0]:4.3f}  |      {abs(atmosphere_table[i, 1] - int(gases[i, 2])):.7f}      |{atmosphere_table[i, 2]:>10.3f}')
        if i < 12:
            outfile.write('\n')
    outfile.close()




'''
C. Model the Atmosphere
'''

def atmosphere_model(mu, h, rho0, T0, M, R):  

    '''
    function for modelling the atmosphere using
    our derived expressions for the density and
    temperature
    '''

    N = len(h)
    rho = np.zeros(N)
    T = np.zeros(N)

    dh = h[1]
    rho[0] = rho0
    T[0] = T0

    def g(altitude):     

        ''' the gravitational acceleration at a given altitude '''

        return G*M/(R + altitude)**2

    for i in range(N - 1):
        rho[i+1] = rho[i]*(1 - mu*mH*g(h[i])*dh/(T[i]*k_B*gamma))

        ''' checking if we're within the adiabatic or the isothermal layer '''

        if T[i] > T0/2:                             
            T[i+1] = T[i] - mu*mH*g(h[i])*(gamma - 1)/(k_B*gamma)*dh
        else:
            T[i+1] = T0/2 

    return rho, T


mu = 29.13                                  # mean mass of a particle in the atmosphere [kg]
T0 = 266.93                                 # approximation for Buttercup's surface temperature found in part 3 [K]
rho0 = system.atmospheric_densities[3]      # the density of Buttercup's atmosphere at surface level [kg/m^3]
M = system.masses[3]*m_sun                  # Buttercups mass [kg]
R = system.radii[3]*1e3                     # Buttercups radius [m]
h = np.linspace(0, 100000, int(1e6)) 
rhoarray, Tarray = atmosphere_model(mu, h, rho0, T0, M, R)

fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.plot(h, rhoarray, color = 'hotpink')
ax1.set_xlabel(r'altitude ($h$) $[m]$')
ax1.set_ylabel(r'density ($\rho$) $[\frac{kg}{m^3}]$')
ax2.plot(h, Tarray, color = 'orange')
ax2.set_xlabel(r'altitude ($h$) $[m]$')
ax2.set_ylabel(r'temperature ($T$) $[K]$')
fig.suptitle(r'The density $\rho$ and temperature $T$ of the atmosphere' + '\n' + 'on Buttercup as a function of altitude $h$')
plt.show()
fig.savefig('density&temperature.pdf')

rho = interp1d(h, rhoarray, fill_value='extrapolate')  
T = interp1d(h, Tarray, fill_value='extrapolate')   