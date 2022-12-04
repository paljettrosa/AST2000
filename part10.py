#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.star_population import StarPopulation

utils.check_for_newer_version()
system = SolarSystem(83856)
stars = StarPopulation(seed=83856)
plt.rcParams.update({'font.size': 14})

G = const.G                         # gravitation constant [m^3/s^2/kg^1]
σ = const.sigma                     # Stefan-Boltzmann constant [W/m^2/K^4]
k_B = const.k_B                     # Boltzmann constant [m^2*kg/s^2/K]
h = 6.62607015*1e-34                # Planck's constant [Js]
ly = 9.4605284*1e15                 # one lightyear [m]
u = 1.661e-27                       # atomic unit of mass [kg]
me = 9.1093837*1e-31                # mass of an electron [kg]
mH = 1.00784*u                      # mass of a hydrogen atom [kg]
mHe = 4.002602*u                    # mass of a helium atom [kg]
solar_M = const.m_sun               # one solar mass [kg]
solar_R = const.R_sun               # one solar radius [m]
solar_L = const.L_sun               # one solar luminosity [W]


'''
1A. Main Sequence Star
'''

M_sun = system.star_mass*solar_M                # mass of our sun [kg]
T_sun = system.star_temperature                 # surface temperature of our sun [K]
R_sun = system.star_radius*1e3                  # radius of our sun [m]
L_sun = σ*T_sun**4*4*np.pi*R_sun**2             # luminosity of our sun [W]

T = stars.temperatures 
L = stars.luminosities 
R = stars.radii   

c_sun = np.array([system.star_color])/255                                       # color of our sun
s_sun = np.maximum(1e3*(R_sun/solar_R - R.min())/(R.max() - R.min()), 1.0)      # relative size of our sun

s_earthsun = np.maximum(1e3*(1.0 - R.min())/(R.max() - R.min()), 1.0)           # relative size of the Earth's sun

c = stars.colors
s = np.maximum(1e3*(R - R.min())/(R.max() - R.min()), 1.0) 

fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='k', marker='*', s=s_sun*80, linewidth=0.01, label='our sun')
ax.scatter(5778, 1.0, c='k', marker='.', s=s_earthsun*80, linewidth=0.01, label="Earth's sun")
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram.pdf')

t_life = (0.007*0.1*M_sun*const.c**2)/(L_sun*3600*24*365)     # expected life time of our sun [h]

''' finding proportionality constants '''

κ_sun = T_sun*R_sun/M_sun
χ_sun = L_sun/M_sun**4

T_earthsun = 5778                               # temperature of the Earth's sun [K]
κ_earthsun = T_earthsun*solar_R/solar_M        
χ_earthsun = solar_L/solar_M**4  



'''
1B. Giant Molecular Cloud
'''

T_GMC = 10                                      # temperature of the GMC [K]
μ = (0.75*mH + 0.25*mHe)/mH                     # mean molecular mass of the GMC
R_GMC = G*M_sun*μ*mH/(5*k_B*T_GMC)              # radius of the GMC [m]
L_GMC = σ*T_GMC**4*4*np.pi*R_GMC**2             # luminosity of the GMC [W]

s_GMC = np.maximum(1e3*(R_GMC/solar_R - R.min())/(R.max() - R.min()), 1.0)      # relative size of the GMC 

fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='k', marker='*', s=s_sun*80, linewidth=0.01, label='our sun')
ax.scatter(5778, 1.0, c='k', marker='.', s=s_earthsun*80, linewidth=0.01, label="Earth's sun")
ax.scatter(T_GMC, L_GMC/solar_L, c='k', marker='x', s=s_GMC/1000, label="Giant Molecular Cloud")
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 3000, 10])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_GMC.pdf')



'''
2A. Core Temperature
'''
μ = 1                                                   # the mean molecular mass of our star
ρ_0 = M_sun/(4/3*np.pi*R_sun**3)                        # the density of our star [kg/m^3]

T_c = T_sun + (2*np.pi/3)*G*ρ_0*(μ*mH/k_B)*R_sun**2     # the core temperature of our star [K]



'''
2B. Energy Production and Luminosity
'''
ϵ_0pp = 1.08*1e-12                                      # amount of energy released per unit in the pp-chain [Wm^3/kg^2]
ϵ_0CNO = 8.24*1e-31                                     # amount of energy released per unit in the CNO-cycle [Wm^3/kg^2]

X_H = 0.745                                             # mass fraction of Hydrogen
X_CNO = 0.002                                           # mass fraction of Carbon, Nitrogen and Oxygen

ϵ_pp = ϵ_0pp*X_H**2*ρ_0*(T_c/1e6)**4                    # reaction rate of pp-chain [W/kg^2]
ϵ_CNO = ϵ_0CNO*X_H*X_CNO*ρ_0*(T_c/1e6)**20              # reaction rate of CNO-cycle [W/kg^2]

L_approx = 0.008*(ϵ_pp + ϵ_CNO)*M_sun                   # approximate luminosity of our star based on core reactions [W]
T_approx = (L_approx/(σ*4*np.pi*0.4*R_sun**2))**(1/4)   # approximate temperature of our star based on core reactions [K]




'''
3A. Leaving the Main Sequence
'''
fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(5500, L_sun/solar_L+20, c='orange', marker='*', s=s_sun*120, edgecolor='k', linewidth=0.01, label='sub giant')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_subgiant.pdf')


fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(2500, L_sun/solar_L+50, c='salmon', marker='*', s=s_sun*160, edgecolor='k', linewidth=0.01, label='red giant')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_criticallylowtemp.pdf')


fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(2500, L_sun/solar_L+200, c='salmon', marker='*', s=s_sun*220, edgecolor='k', linewidth=0.01, label='red giant')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_horizontalbranchright.pdf')


fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(5500, L_sun/solar_L+50, c='orange', marker='*', s=s_sun*180, edgecolor='k', linewidth=0.01, label='red giant')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_horizontalbranchleft.pdf')


fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(3500, 10**5, c='coral', marker='*', s=s_sun*340, edgecolor='k', linewidth=0.01, label='red giant')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_asymptoticgiantbranch.pdf')


fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(25000, 3*10**5, c='pink', marker='o', s=s_sun*340, edgecolor='k', linewidth=0.01, label='red giant surrounded by nebulae')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_beforewhitedwarf.pdf')


fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='coral', linewidth=0.0001, zorder=1)
ax.scatter(T_sun, L_sun/solar_L, c='gold', marker='*', s=s_sun*180, edgecolor='k', linewidth=0.01, label='our sun today')
ax.scatter(30000, 10**(-1.4), c='lightcyan', marker='*', s=s_sun*80, edgecolor='k', linewidth=0.001, label='white dwarf')
ax.legend()

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('HR_diagram_whitedwarf.pdf')



'''
3B. The End
'''
M_WD = M_sun/(8*solar_M)*1.4                                                            # mass of the white dwarf [M]
R_WD = (3/(2*np.pi))**(4/3)*h**2/(20*me*G)*(1/(2*mH))**(5/3)*(M_WD*solar_M)**(-1/3)     # radius of the white dwarf [m]

ρ_WD = M_WD*solar_M/(4/3*np.pi*R_WD**3)         # density of the white dwarf [kg/m^3]
one_litre = ρ_WD*0.001                          # mass of one litre white dwarf material [kg]

g_WD = - G*M_WD*solar_M/(R_WD**2)               # gravitational acceleration at the surface of the white dwarf [m/s^2]