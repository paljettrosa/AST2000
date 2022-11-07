#EGEN KODE
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts 

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [10978]) 

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
checking our spacecraft's velocity relative to the destination planet 
to make sure it's not higher than the upper boundary of 10 km/s
'''
#TODO THEORY - fiks disse utledningene

#TODO utled et uttrykk for den maksimale doppler shiften som raketten isåfall kan observere (oppg 1)

#TODO absvel = np.linalg.norm(np.array([vx, vy]))
#TODO print(absvel), skal vi bruke denne noe sted?

#TODO utled et uttrykk for standard deviation (oppg 2)

'''
Downloading the Data
'''

spectrum_data = []
with open(r'/Users/paljettrosa/Documents/AST2000/spectrum_seed56_600nm_3000nm.txt', 'r') as infile:
    for line in infile:
        spectrum_data.append([float(line.split()[0]), float(line.split()[1])])
    infile.close()
spectrum_data = np.array(spectrum_data)

noise_data = []
with open(r'/Users/paljettrosa/Documents/AST2000/sigma_noise.txt', 'r') as infile:
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
    #TODO formelen for dlambda må utledes i A1
    dlambda = (vmax + np.sqrt(k_B*Tmax/m))*lambda0/c            # maximum possible Doppler shift that the spacecraft can observe for this element's spectral line at this wavelength
    #TODO sjekk at denne slicingen blir riktig
    tol = 1e-3
    lower_idx = np.where(abs(data[:, 0] - np.round(lambda0 - dlambda, 3)) < tol)[0][0]      
    upper_idx = np.where(abs(data[:, 0] - np.round(lambda0 + dlambda, 3)) < tol)[0][-1]    
    sliced_data = data[lower_idx:upper_idx]                                 
    sliced_data[:, 0] = sliced_data[:, 0]*1e-9                    # changing the wavelength units to meters

    #TODO endre disse?
    Fmin = np.linspace(0.7, 1, 50)
    lambdas = np.linspace(sliced_data[0, 0], sliced_data[-1, 0], 50)
    T = np.linspace(150, 450, 50)

    def sigma(Ti):

        ''' standard deviation formula '''
        '''
        TODO
        lambda0 er spektrallinjen til gassen uten dopplershift?
        m er massen til et gassmolekyl og T er temperaturen til gassen?
        UTLED FORMELEN
        '''
        return sliced_data[:, 0]/c*np.sqrt(k_B*Ti/m)
    
    def gaussian_lineprofile(Fmini, lambdai, Ti):

        ''' gaussian line profile formula for modelling spectral lines '''
        '''
        TODO 
        Gaussisk fordeling ved gaussisk linjeprofilformel
        Skal denne formelen utledes? eller finnes den i forelesningsnotatene 1D?
        '''
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

    plt.plot(sliced_data[:, 0], sliced_data[:, 1], color = color1, label = 'measured data')
    plt.plot(sliced_data[:, 0], gaussian_lineprofile(best_Fmin, best_lambda, best_T), color = color2, label = 'gaussian line profile')
    plt.legend()
    plt.xlabel("wavelength [nm]")
    plt.ylabel("flux")
    plt.title(f"The Chisquare approximation for {gas}'s" + '\n' + fr"spectral line at $\lambda_0$ = {lambda0} nm")
    fig = plt.gcf()
    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/chisquare_{gas}_{lambda0}.pdf')

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

with open(r'/Users/paljettrosa/Documents/AST2000/atmosphere_table.txt', 'w') as outfile:
    outfile.write(' Element & Wavelength |   Flux   |    Doppler Shift    |  Temperature  ' + '\n')
    outfile.write('______________________|__________|_____________________|_______________' + '\n')
    for i in range(12):
        atmosphere_table[i] = chisquare_method(str(gases[i, 0]), float(gases[i, 1]), int(gases[i, 2]), str(gases[i, 3]), str(gases[i, 4]))
        outfile.write(f'      {gases[i, 0]:>3} - {str(gases[i, 2]):4}      |   {atmosphere_table[i, 0]:4.3f}  |      {abs(atmosphere_table[i, 1] - int(gases[i, 2])):.7f}      |{atmosphere_table[i, 2]:>10.3f}')
        if i < 12:
            outfile.write('\n')
    outfile.close()

'''
#TODO THEORY

Filtering the Real Lines from the Flukes & The Mean Molecular Weight

Looking at our table of each compound's spectral lines and their associated flux, doppler shifts
and temperatures, there were five that especially stood out.

 Element & Wavelength |   Flux   |    Doppler Shift    |  Temperature  
______________________|__________|_____________________|_______________
      H2O - 720       |   0.829  |      0.0204147      |   382.653
      H2O - 820       |   0.737  |      0.0235265      |   192.857
      CH4 - 2200      |   0.737  |      0.0617327      |   309.184
       CO - 2340      |   0.896  |      0.0649265      |   303.061
      N2O - 2870      |   0.878  |      0.0794224      |   247.959

The two H2O lines have very similar Doppler shifts and rather low Fmin's, but if these are real, the 
other spectral lines are probably flukes, as they have much higher Doppler shifts. The biggest problem with
the H2O-lines is that they have very different temperatures, both of with in addition to this differ a lot
from Buttercup's surface temperature, which is approximately 266.93 K. Since they have such different 
temperatures, one of them has to be a fluke. In this case, we have only found one spectral line which is real.
The probability of our atmosphere only consisting of H2O is very slim, so we take a look at the three next
spectral lines instead. These have very similar Doppler shifts, Fmin's that are not too high, and temperatures
that don't differ too much from each other. In addition, these temperatures don't differ too much from the
surface temperature either. N2O's Doppler shift is a little bigger than the others', but this may be because
of the large amount of noise in the data. 

We also automatically ruled out all the elements with temperatures exactly 150 and 450 as flukes, as 
these most likely are just the best approfimations for a potential spectral line when no other 
candidates were better. An actual spectral line for a compound with spectral line exactly 150K for 
example, would probably have an even lower temperature (TODO riktig?)

We assume that only the CH4 - 2200, CO - 2340 and N2O - 2870 lines are real, and our atmosphere therefore consists
of 33.3% methane, 33.3% carbon monoxide, and 33.3% nitrogen dioxide. The fact that the atmosphere consists of CH4 and
NO2 tells us that there is a probability of life existing on our destination planet. This chance is weakened by the 
presence of CO however, as this gas is very poisonous to humans, and probably to many other life sources as well.

The mean molecular weight is given by the formula mu = sum(f_i*m_i/mH), where i goes from 1 to N, and N is the number of
gases in the atmosphere. In our case, N is 3. f_i is the fraction of atmosphere which consists of the ith gas, m_i is the 
mass of this gas, and mH is the mass of the hydrogen atom. This means that the unit of the mean molecular weight is in
number of hydrogen atoms. Our mean molecular weight is:
    mu = 1/3*(12.0107 + 4*1.00784)/1.00784 + 1/3*(12.0107 + 15.999)/1.00784 + 1/3*(2*14.0067 + 15.999)/1.00784
       = 1/3*16.04206/1.00784 + 1/3*28.0067/1.00784 + 1/3*44.0124/1.00784
       = 5.3057562 + 9.26294518 + 14.5566757
       ≈ 29.13   TODO riktig?
'''



'''
C. Model the Atmosphere
'''

'''
#TODO THEORY

We know that while T > T0/2, the equation p^(1 - gamma)*T^gamma = C for adiabatic gases holds. p denotes the 
atmospheric pressure. Using this equation to find out how the temperature decreases with altitude, we want to 
take the derivative of this equation with respect to the altitude h (as the atmosphere is spherically symmetrical)
This gives us (1 - gamma)*p^(- gamma)*T^gamma + p^(1 -gamma)*T^(gamma - 1) = 0:
      dpdh = p(h)*gamma*dTdh/(T(h)*(gamma - 1))                                                       (1)
           = - rho(h)*g(h)                                                                            (2), follows from the assumption of hydrostatic equilibrium
From the assumption of the atmosphere being an ideal gas, we also have: 
      p(h) = rho(h)*k_B*T(h)/mu                                                                       (3)
Which gives us:
    rho(h) = p(h)*mu/(k_B*T(h))                                                                       (4)
Adding this expression for rho(h) to (2), we get:
      dpdh = - p(h)*mu*g(h)/(k_B*T(h))                                                                (5)
Comparing this to (1), we can derive an expression for dTdh:
    p(h)*gamma*dTdh/(T(h)*(gamma - 1)) = - p(h)*mu*g(h)/(k_B*T(h))                                    (6)
                                  dTdh = - p(h)*mu*g(h)/(k_B*T(h)) * (T(h)*(gamma - 1))/p(h)*gamma    (7)
                                       = - mu*g(h)*(gamma - 1)/(k_B*gamma)                            
This gives us an expression for T(h + dh):
 T(h + dh) = T(h) + dTdh*dh                                                                      
           = T(h) - mu*g(h)*(gamma - 1)/(k_B*gamma)*dh                                                (8)

To find the density rho as a function of altitude h, we take a look at (2) and (3):
                         dpdh = d(rho(h)*k_B*T(h)/mu)dh,                                              follows from (3)
      d(rho(h)*k_B*T(h)/mu)dh = - rho(h)*g(h),                                                        follows from (2)
             d(rho(h)*T(h))dh = - rho(h)*g(h)*mu/k_B                                                  (9)
    drhodh*T(h) + rho(h)*dTdh = - rho(h)*g(h)*mu/k_B,                                                 product rule
We can rearrange this to find an expression for drhodh:
                  drhodh*T(h) = - rho(h)*(g(h)*mu/k_B + dTdh)
                       drhodh = - rho(h)/T(h)*(g(h)*mu/k_B + dTdh)
                              = - rho(h)*mu*g(h)*(1 - (gamma - 1)/gamma)/(T(h)*k_B)
                              = - rho(h)*mu*g(h)/(T(h)*k_B*gamma)                                     (10)
This gives us the following expression for rho(h + dh):
                  rho(h + dh) = rho(h) + drhodh*dh
                              = rho(h) - rho(h)*mu*g(h)/(T(h)*k_B*gamma)*dh 
                              = rho(h)*(1 - mu*g(h)*dh/(T(h)*k_B*gamma))                              (11)    
'''

def atmosphere_model(mu, h, rho0, T0, M, R):  
    N = len(h)
    rho = np.zeros(N)
    T = np.zeros(N)

    dh = h[1]
    rho[0] = rho0
    T[0] = T0

    def g(height):                                # the gravitational acceleration as a function of height [m/s^2]
        return G*M/(R + height)**2
    #TODO eller skal vi bare bruke g ved overflaten?

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
fig.savefig(f'/Users/paljettrosa/Documents/AST2000/density&temperature.pdf')

rho = interp1d(h, rhoarray, fill_value='extrapolate')  
T = interp1d(h, Tarray, fill_value='extrapolate')   