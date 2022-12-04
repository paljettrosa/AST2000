#EGEN KODE
#KANDIDATER 15361 & 15384
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from part6AD import *

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)
shortcut = SpaceMissionShortcuts(mission, [10978]) 
plt.rcParams.update({'font.size': 12})

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

G = const.G                                     # gravitation constant [m^3/(s^2*kg)]
m_sun = const.m_sun                             # solar mass [kg]
AU = const.AU                                   # one astronomical unit [m]
day = const.day                                 # one day [s]
yr = const.yr                                   # one year [s]

M = system.masses[3]*m_sun                      # Buttercup's mass [kg]
R = system.radii[3]*1e3                         # Buttercup's radiis [m]
T = system.rotational_periods[3]*day            # Buttercup's rotational period [s]
rho0 = system.atmospheric_densities[3]          # the density of the atmosphere on Buttercup by the surface [kg/m^3]
Cd = 1                                          # the drag coefficient, set to 1 for simplicity

m_sc = final_m                                  # the total mass of the spacecraft after the travel was completed [kg]
m_l = mission.lander_mass                       # the mass of the spacecraft's lander [kg]
A_sc = mission.spacecraft_area                  # the cross-sectional area of the spacecraft[m^2]
A_l = mission.lander_area                       # the cross-sectional area of the spacecraft's lander [m^2]

''' code from part 6 '''

gamma = 1.4                         # the heat capacity ratio
k_B = const.k_B                     # Boltzmann constant [m^2*kg/s^2/K]
u = 1.661e-27                       # atomic unit of mass [kg]
m_CH4 = (12.0107 + 4*1.00784)*u     # mass of CH4 molecule [kg]
m_CO = (12.0107 + 15.999)*u         # mass of CO molecule [kg]
m_N2O = (2*14.0067 + 15.999)*u      # mass of N2O molecule [kg]

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

    def g(height):                                # the gravitational acceleration as a function of height [m/s^2]
        return G*M/(R + height)**2

    for i in range(N - 1):
        rho[i+1] = rho[i]*(1 - mu*g(h[i])*dh/(T[i]*k_B*gamma))

        ''' checking if we're within the adiabatic or the isothermal layer '''

        if T[i] > T0/2:                             
            T[i+1] = T[i] - mu*g(h[i])*(gamma - 1)/(k_B*gamma)*dh
        else:
            T[i+1] = T0/2 

    return rho, T

mu = (m_CH4 + m_CO + m_N2O)/3               # mean mass of a particle in the atmosphere [kg]
T0 = 266.93                                 # approximation for Buttercup's surface temperature found in part 3 [K]
h = np.linspace(0, 100000, int(1e6)) 
rhoarray, temparray = atmosphere_model(mu, h, rho0, T0, M, R)

rho = interp1d(h, rhoarray, fill_value='extrapolate')  
temp = interp1d(h, temparray, fill_value='extrapolate') 





'''
A. Air Resistance
'''

def air_resistance(ri, A, vi):

    '''
    function for calculating air resistance
    as a function of altitude
    '''

    w = ri*2*np.pi/T*np.array([0, 1, 0])
    v_drag = np.linalg.norm(vi - w)
    if abs(rho(ri)) >= rho0:
        Fd = 1/2*rho0*Cd*A*v_drag**2
    elif rho(ri) < 1e-4:
        Fd = 0
    else:
        Fd = 1/2*rho(ri)*Cd*A*v_drag**2
    return Fd

vsafe = 3
A = 2*G*m_l*M/(R**2*rho0*Cd*vsafe**2)





'''
B. Simulating the Landing
'''

def simulate_landing(N, dt, t0, r0, v0, vsafe, m_sc, m_l, A_sc, A_l, A_p, vl, launch_pos, deploy_pos, thrust_pos):

    '''
    function for simulating the landing process
    '''

    r = np.zeros((N, 3))
    v = np.zeros((N, 3))

    r[0] = r0
    v[0] = v0

    def gravity(m, ri):
        
        ''' 
        the gravitational force working on us as a function of 
        distance from the Buttercup's center of mass 
        '''

        return np.array([- G*m*M/(R + ri)**2, 0, 0])
    
    def v_terminal(m, ri, A):

        ''' 
        the terminal velocity as a function of altitude 
        '''

        if abs(rho(ri)) > rho0:
            return np.sqrt(2*abs(gravity(m, ri)[0])/(m*rho0*A*Cd))
        elif rho(ri) < 0:
            return 0
        else:
            return np.sqrt(2*abs(gravity(m, ri)[0])/(m*rho(ri)*A*Cd))

    def thrust(A, vt):      

        '''
        the landing thrusters' thrust force as a 
        function of terminal velocity
        '''    

        Fl = np.array([1/2*rho0*A*(vt**2 - vsafe**2), 0, 0])
        if Fl[0] < 0:
            return - Fl
        else:
            return Fl
    
    entered = False
    launched = False
    deployed = False
    thrusted = False
    landed = False
    tol = 5
    for i in range(N-1):
        if abs(v[i, 1]) <= 1e-6:
            Fd_direction = np.array([1, 0, 0])
        else:
            Fd_direction = - v[i]/np.linalg.norm(v[i])
        
        ''' checking if we've entered the atmosphere '''

        if rho(r[i, 0]) > 0 and entered == False:
            t_entered = i
            print(f"Entering the atmosphere {(t_entered*dt)/3600:.2f} hours after starting the simulation")
            print(f"The spacecraft's radial velocity at the moment it enters is {v[i, 0]:.3f} m/s")
            print(f"The spacecraft's azimuthal velocity at the moment it enters is {v[i, 1]*180/np.pi:.3f} deg/s\n")
            entered = True


        ''' checking if we're ready to launch the lander '''

        if abs(r[i, 0] - launch_pos) < tol and launched == False:
            t_launched = i
            v[i] = v[i] + vl
            print(f"Launching the lander {(t_launched*dt)/3600:.2f} hours after starting the simulation")
            print(f"The lander's radial velocity at the moment it is launched is {v[i, 0]:.3f} m/s")
            print(f"The lander's azimuthal velocity at the moment it is launched is {v[i, 1]*180/np.pi:.3f} deg/s\n")
            launched = True
        

        ''' checking if we're ready to deploy our parachute '''

        if abs(r[i, 0] - deploy_pos) < tol*10 and deployed == False:
            t_deployed = i
            if rho(r[i, 0]) <= 0:
                print("You're deploying your parachute too high up in the atmosphere!")
                break
            else:
                print(f"Deploying the parachute {(t_deployed*dt)/3600:.2f} hours after starting the simulation")
                print(f"The lander's radial velocity at the moment we deploy the parachute is {v[i, 0]:.3f} m/s\n")
                deployed = True

        
        ''' in case the lander is still attached to the rocket '''

        if launched == False:
            if abs(v[i, 0]) <= v_terminal(m_sc, r[i, 0], A_sc) and v_terminal(m_sc, r[i, 0], A_sc) != 0:
                v[i, 0] = - v_terminal(m_sc, r[i, 0], A_sc)
                v[i+1] = v[i]
                r[i+1] = r[i] + v[i+1]*dt

            else:
                a = (air_resistance(r[i, 0], A_sc, v[i])*Fd_direction + gravity(m_sc, r[i, 0]))/m_sc
                v[i+1] = v[i] + a*dt/2
                r[i+1] = r[i] + v[i+1]*dt

                if abs(v[i+1, 1]) <= 1e-6:
                    Fd_direction = np.array([1, 0, 0])
                else:
                    Fd_direction = - v[i+1]/np.linalg.norm(v[i+1])

                a = (air_resistance(r[i+1, 0], A_sc, v[i+1])*Fd_direction + gravity(m_sc, r[i+1, 0]))/m_sc
                v[i+1] = v[i+1] + a*dt/2

        
        ''' in case we've launched our lander '''

        if launched == True and deployed == False:
            if abs(v[i, 0]) >= v_terminal(m_l, r[i, 0], A_l):
                v[i, 0] = - v_terminal(m_l, r[i, 0], A_l)
                v[i+1] = v[i]
                r[i+1] = r[i] + v[i+1]*dt

            else:
                Fd = air_resistance(r[i, 0], A_l, v[i])
                if Fd/A_l >= 1e7:
                    print('Our lander was incinerated!')
                    break
                else:
                    a = (Fd*Fd_direction + gravity(m_l, r[i, 0]))/m_l
                    v[i+1] = v[i] + a*dt/2
                    r[i+1] = r[i] + v[i+1]*dt

                if abs(v[i+1, 1]) <= 1e-6:
                    Fd_direction = np.array([1, 0, 0])
                else:
                    Fd_direction = - v[i+1]/np.linalg.norm(v[i+1])

                Fd = air_resistance(r[i+1, 0], A_l, v[i+1])
                if Fd/A_l >= 1e7:
                    print('Our lander was incinerated!')
                    break
                else:
                    a = (Fd*Fd_direction + gravity(m_l, r[i+1, 0]))/m_l
                    v[i+1] = v[i+1] + a*dt/2


        ''' in case we've deployed our parachute '''

        if deployed == True and r[i, 0] > thrust_pos:
            if abs(v[i, 0]) >= v_terminal(m_l, r[i, 0], A_p):
                v[i, 0] = - v_terminal(m_l, r[i, 0], A_p)
                v[i+1] = v[i]
                r[i+1] = r[i] + v[i+1]*dt

            else:
                Fd = air_resistance(r[i, 0], A_p, v[i])
                if Fd > 2.5*1e5: 
                    print('The parachute failed!')
                    break
                else:
                    a = (Fd*Fd_direction + gravity(m_l, r[i, 0]))/m_l
                    v[i+1] = v[i] + a*dt/2
                    r[i+1] = r[i] + v[i+1]*dt

                    if abs(v[i+1, 1]) <= 1e-6:
                        Fd_direction = np.array([1, 0, 0])
                    else:
                        Fd_direction = - v[i+1]/np.linalg.norm(v[i+1])

                Fd = air_resistance(r[i+1, 0], A_p, v[i+1])
                if Fd > 2.5*1e5:
                    print('The parachute failed!')
                    break
                else:
                    a = (Fd*Fd_direction + gravity(m_l, r[i+1, 0]))/m_l
                    v[i+1] = v[i+1] + a*dt/2
        

        ''' in case we've started the landing thrusters '''

        if r[i, 0] <= thrust_pos:
            if thrusted == False:
                t_thrusted = i
                vt = v_terminal(m_l, r[i, 0], A_p)
                print(f"Activating the landing thrusters {(t_thrusted*dt)/3600:.2f} hours after starting the simulation")
                print(f"They excel a thrust force of {thrust(A_p, vt)[0]:.1f} N")
                print(f"The lander's radial velocity at the moment the landing thrusters are activated is {v[i, 0]:.3f} m/s\n")
                thrusted = True

            vt = v_terminal(m_l, r[i, 0], A_p)
            if abs(v[i, 0]) <= vsafe:
                a = 0
                v[i+1] = v[i]
            else:
                a = thrust(A_p, vt)/m_l
                v[i+1] = v[i] + a*dt/2
            r[i+1] = r[i] + v[i+1]*dt
            
            vt = v_terminal(m_l, r[i+1, 0], A_p)
            if abs(v[i+1, 0]) <= vsafe:
                a = 0
            else:
                a = thrust(A_p, vt)/m_l
                v[i+1] = v[i] + a*dt/2


            ''' checking if we've reached the surface '''

            if r[i+1, 0] < 1:
                if abs(v[i+1, 0]) > vsafe:
                    print('We crashlanded! Adjust the thrusters')
                else:
                    print("Woohoo! We've softly landed on Buttercup :)")
                    print(f"The simulated landing took approximately {(i+1)*dt/3600:.2f} hours")
                    print(f"The lander's radial velocity at the moment it landed was {v[i+1, 0]:.3f} m/s\n")
                r = r[:i+2]
                v = v[:i+2]
                time_elapsed = (i+1)*dt
                final_time = t0 + time_elapsed
                final_pos = r[-1]
                final_vel = v[-1]
                t = np.linspace(t0, final_time, i+2)
                landed = True
                break

    if landed == False:
        time_elapsed = N*dt
        final_time = t0 + time_elapsed
        final_pos = r[-1]
        final_vel = v[-1]
        t = np.linspace(t0, final_time, N)
    return t, r, v, final_time, final_pos, final_vel, time_elapsed, t_entered, t_launched, t_deployed, t_thrusted
    

N = 100000
dt = 0.1

''' adjusting our position so that we land on the day side '''

coast_time = 0.0002*yr
landing.fall(coast_time)
t0, position, velocity = landing.orient()
r = np.linalg.norm([position[0], position[1], position[2]])
phi = np.arctan(position[1]/position[0])
theta = np.pi/2
r0 = np.array([r - R, phi, theta])

vr = (position[0]*velocity[0] + position[1]*velocity[1])/r
vphi = (position[0]*velocity[1] - position[1]*velocity[0])/r**2 

''' there are no forces working on us in the theta-direction, so we will stay at put at theta = pi/2 '''

vtheta = 0
v0 = np.array([vr, vphi, vtheta])

vl = np.array([-100, 0, 0])           # the velocity we want to boost our lander with, spherical coordinates
launch_pos = 10000
deploy_pos = 500
thrust_pos = 100

t, pos, vel, final_time, final_pos, final_vel, time_elapsed, t_entered, t_launched, t_deployed, t_thrusted = simulate_landing(N, dt, t0, r0, v0, vsafe, m_sc, m_l, A_sc, A_l, A, vl, launch_pos, deploy_pos, thrust_pos)

print('The spherical coordinates of our spacecraft is:')
print("  When we were about to descend:")
print(f'       r = {r0[0]:.1f} meters')
print(f'     phi = {r0[1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f"       t = {t0:.1f} seconds after the module's landing sequence was initiated\n")
print("  By the time we've successfully landed:")
print(f'       r = {final_pos[0]:.1f} meters')
print(f'     phi = {final_pos[1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f"       t = {final_time:.1f} seconds after the module's landing sequence was initiated\n")


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)
scale_x = 60
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/scale_x))
scale_y = 1e3
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))

ax1.plot(t-t0, pos[:, 0], color = 'palevioletred', label = 'radial trajectory', zorder = 1)
ax1.scatter(t[t_entered]-t0, pos[t_entered, 0], color = 'crimson', marker = 'o', s = 20, label = 'entering atmosphere')
ax1.scatter(t[t_launched-1]-t0, pos[t_launched-1, 0], color = 'crimson', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed-1]-t0, pos[t_deployed-1, 0], color = 'crimson', marker = '*', label = 'deploying parachute')
ax1.scatter(t[t_thrusted]-t0, pos[t_thrusted, 0], color = 'crimson', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time elapsed (t) [min]')
ax1.set_ylabel('radial position (r) [km]')
ax1.xaxis.set_major_formatter(ticks_x)
ax1.yaxis.set_major_formatter(ticks_y)

ax2.plot(t-t0, pos[:, 1]*180/np.pi, color = 'violet', label = 'azimuthal trajectory', zorder = 1)
ax2.scatter(t[t_entered]-t0, pos[t_entered, 1]*180/np.pi, color = 'darkmagenta', marker = 'o', s = 20, label = 'entering atmosphere')
ax2.scatter(t[t_launched-1]-t0, pos[t_launched-1, 1]*180/np.pi, color = 'darkmagenta', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed-1]-t0, pos[t_deployed-1, 1]*180/np.pi, color = 'darkmagenta', marker = '*', label = 'deploying parachute')
ax2.scatter(t[t_thrusted]-t0, pos[t_thrusted, 1]*180/np.pi, color = 'darkmagenta', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal position $(\phi)$ [deg]')
ax2.xaxis.set_major_formatter(ticks_x)

fig.suptitle("Radial and azimuthal position during the entire landing")
plt.show()
fig.savefig('landing_trajectory.pdf')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t[t_launched-100:]-t0, pos[t_launched-100:, 0], color = 'hotpink', label = 'radial trajectory', zorder = 1)
ax1.scatter(t[t_launched-1]-t0, pos[t_launched-1, 0], color = 'deeppink', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed-1]-t0, pos[t_deployed-1, 0], color = 'deeppink', marker = '*', label = 'deploying parachute') 
ax1.scatter(t[t_thrusted]-t0, pos[t_thrusted, 0], color = 'deeppink', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time elapsed (t) [min]')
ax1.set_ylabel('radial position (r) [m]')
ax1.xaxis.set_major_formatter(ticks_x)

ax2.plot(t[t_launched-100:]-t0, pos[t_launched-100:, 1]*180/np.pi, color = 'plum', label = 'azimuthal trajectory', zorder = 1)
ax2.scatter(t[t_launched-1]-t0, pos[t_launched-1, 1]*180/np.pi, color = 'mediumorchid', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed-1]-t0, pos[t_deployed-1, 1]*180/np.pi, color = 'mediumorchid', marker = '*', label = 'deploying parachute') 
ax2.scatter(t[t_thrusted]-t0, pos[t_thrusted, 1]*180/np.pi, color = 'mediumorchid', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal position $(\phi)$ [deg]')
ax2.xaxis.set_major_formatter(ticks_x)

fig.suptitle("The lander's trajectory after entering the atmosphere")
plt.show()
fig.savefig('landing_trajectory_closeups.pdf')



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t-t0, vel[:, 0], color = 'palevioletred', label = 'radial velocity', zorder = 1)
ax1.scatter(t[t_entered]-t0, vel[t_entered, 0], color = 'crimson', marker = 'o', s = 20, label = 'entering atmosphere')
ax1.scatter(t[t_launched-1]-t0, vel[t_launched-1, 0], color = 'crimson', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed-1]-t0, vel[t_deployed-1, 0], color = 'crimson', marker = '*', label = 'deploying parachute')
ax1.scatter(t[t_thrusted]-t0, vel[t_thrusted, 0], color = 'crimson', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time elapsed (t) [min]')
ax1.set_ylabel(r'radial velocity $(v_r)$ $[\frac{km}{s}]$')
ax1.xaxis.set_major_formatter(ticks_x)
ax1.yaxis.set_major_formatter(ticks_y)

ax2.plot(t-t0, vel[:, 1]*180/np.pi, color = 'violet', label = 'azimuthal velocity', zorder = 1)
ax2.scatter(t[t_entered]-t0, vel[t_entered, 1]*180/np.pi, color = 'darkmagenta', marker = 'o', s = 20, label = 'entering atmosphere')
ax2.scatter(t[t_launched-1]-t0, vel[t_launched-1, 1]*180/np.pi, color = 'darkmagenta', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed-1]-t0, vel[t_deployed-1, 1]*180/np.pi, color = 'darkmagenta', marker = '*', label = 'deploying parachute')
ax2.scatter(t[t_thrusted]-t0, vel[t_thrusted, 1]*180/np.pi, color = 'darkmagenta', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal velocity $(v_{\phi})$ $[\frac{deg}{s}]$')
ax2.xaxis.set_major_formatter(ticks_x)

fig.suptitle("Radial and azimuthal velocity during the entire landing")
plt.show()
fig.savefig('landing_velocity.pdf')



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t[t_launched-100:]-t0, vel[t_launched-100:, 0], color = 'hotpink', label = 'radial velocity', zorder = 1)
ax1.scatter(t[t_launched-1]-t0, vel[t_launched-1, 0], color = 'deeppink', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed-1]-t0, vel[t_deployed-1, 0], color = 'deeppink', marker = '*', label = 'deploying parachute') 
ax1.scatter(t[t_thrusted]-t0, vel[t_thrusted, 0], color = 'deeppink', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time elapsed (t) [min]')
ax1.set_ylabel(r'radial velocity $(v_r)$ $[\frac{m}{s}]$')
ax1.xaxis.set_major_formatter(ticks_x)

ax2.plot(t[t_launched-100:]-t0, vel[t_launched-100:, 1]*180/np.pi, color = 'plum', label = 'azimuthal velocity', zorder = 1)
ax2.scatter(t[t_launched-1]-t0, vel[t_launched-1, 1]*180/np.pi, color = 'mediumorchid', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed-1]-t0, vel[t_deployed-1, 1]*180/np.pi, color = 'mediumorchid', marker = '*', label = 'deploying parachute') 
ax2.scatter(t[t_thrusted]-t0, vel[t_thrusted, 1]*180/np.pi, color = 'mediumorchid', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal velocity $(v_{\phi})$ $[\frac{deg}{s}]$')
ax2.xaxis.set_major_formatter(ticks_x)

fig.suptitle("The lander's velocity after entering the atmosphere")
plt.show()
fig.savefig('landing_velocity_closeups.pdf')






'''
C. Landing our Spacecraft
'''

''' we use the middle of the planet on the picture 'scouting8.png' as the landing site '''

print('The spherical coordinates of our chosen landing site is:')
print('  At the moment when we photographed it:')
print(f'       r = {coords[7, 0]:.1f} meters')
print(f'     phi = {coords[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f"       t = {coords[7, 3]:.1f} seconds after the module's landing sequence was initiated\n")
coords_afterlastpic = new_coords(coords, 0)
print("  After we took the last photograph of Buttercup:")
print(f'       r = {coords_afterlastpic[7, 0]:.1f} meters')
print(f'     phi = {coords_afterlastpic[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f"       t = {coords_afterlastpic[7, 3]:.1f} seconds after the module's landing sequence was initiated\n")
coords_aftercoast = new_coords(coords_afterlastpic, coast_time)
print("  When we're ready to descend:")
print(f'       r = {coords_aftercoast[7, 0]:.1f} meters')
print(f'     phi = {coords_aftercoast[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f"       t = {coords_aftercoast[7, 3]:.1f} seconds after the module's landing sequence was initiated\n")
coords_whenlanded = new_coords(coords_aftercoast, time_elapsed)
print("  By the time we've successfully landed:")
print(f'       r = {coords_whenlanded[7, 0]:.1f} meters')
print(f'     phi = {coords_whenlanded[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f"       t = {coords_whenlanded[7, 3]:.1f} seconds after the module's landing sequence was initiated\n")                        

landing_site = np.array([coords_whenlanded[7, 0]*np.cos(coords_whenlanded[7, 1]), coords_whenlanded[7, 0]*np.sin(coords_whenlanded[7, 1]), 0])
final_pos_cartesian = np.array([final_pos[0]*np.cos(final_pos[1]), final_pos[0]*np.sin(final_pos[1]), 0])
distance_from_site = final_pos_cartesian - landing_site     # how far off we are from our chosen landing site
vboost = - distance_from_site/time_elapsed                  # how much we need to boost our rocket to land on the correct spot


''' slowing down '''

landing.start_video()
landing.boost(- velocity/3)                         
start_time, position, velocity = landing.orient()
landing.fall_until_time(t0 + t_launched*dt)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {(time-start_time)/3600:.2f} hours after we initiated the landing is {r:.1f} m\n')

''' launching lander '''

landing.launch_lander(vboost)
landing.adjust_parachute_area(A)
landing.fall_until_time(t0 + t_deployed*dt)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {(time-start_time)/3600:.2f} hours after we initiated the landing is {r:.1f} m\n')

''' deploying parachute and adjusting landing thrusters '''

landing.deploy_parachute()
vt = np.sqrt(2*G*M/((R + 500)**2*rho(100)*A*Cd))
Fl = 1/2*rho0*A*(vt**2 - vsafe**2)
if Fl < 0:
    landing.adjust_landing_thruster(- Fl, 50)
else:
    landing.adjust_landing_thruster(Fl, 50)

landing.look_in_direction_of_motion(relative_polar_angle=np.pi/2, relative_azimuth_angle=0)
landing.fall_until_time(t0 + t_thrusted*dt)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {(time-start_time)/3600:.2f} hours after we initiated the landing is {r:.1f} m\n')

landing.fall_until_time(final_time + 2850)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {(time-start_time)/3600:.2f} hours after we initiated the landing is {r:.1f} m\n')

landing.activate_landing_thruster()
landing.fall_until_time(final_time + 3200)
time, position, velocity = landing.orient()

if landing.reached_surface == True:
    site_pos = new_coords(coords, time - t0)
    new_landing_site_coordinates = np.array([site_pos[7, 0]*np.cos(site_pos[7, 1]), site_pos[7, 0]*np.sin(site_pos[7, 1]), 0])
    distance_from_site = np.linalg.norm(position - new_landing_site_coordinates)
    print('\n')
    print(f"We reached Buttercup's surface {(time-start_time)/3600:.2f} hours after we initiated the landing!")
    print(f"Our landing position coordinates is ({position[0]*1e-3:.1f}, {position[1]*1e-3:.1f}, 0.0) km")  
    print(f"Our chosen landing site's current position ({new_landing_site_coordinates[0]*1e-3:.1f}, {new_landing_site_coordinates[1]*1e-3:.1f}, 0.0) km")
    print(f"Our chosen landing site's azimuthal position has changed to {site_pos[7, 1]*180/np.pi:.4f} degrees")
    print(f"We landed {distance_from_site*1e-3:.1f} km away from the planned landing site\n")

landing.finish_video('landing.xml')




'''
RESULTS:
        FROM SIUMULATION:
    Launching the lander 0.90 hours after starting the simulation
    The lander's radial velocity at the moment it is launched is -175.035 m/s
    The lander's azimuthal velocity at the moment it is launched is 0.000 deg/s

    Deploying the parachute 1.23 hours after starting the simulation
    The lander's radial velocity at the moment we deploy the parachute is -6.712 m/s

    Activating the landing thrusters 1.62 hours after starting the simulation
    They excel a thrust force of 677.4 N
    The lander's radial velocity at the moment the landing thrusters are activated is -0.317 m/s

    Woohoo! We've softly landed on Buttercup :)
    The simulated landing took approximately 1.71 hours
    The lander's radial velocity at the moment it landed was -0.317 m/s
    

        COORDINATES:
    The spherical coordinates of our spacecraft is:
        When we were about to descend:
            r = 9305389.4 meters
            phi = -74.0 degrees
            theta = 90.0 degrees
            t = 36290.5 seconds after the module's landing sequence was initiated

        By the time we've successfully landed:
            r = 1.0 meters
            phi = -23.9 degrees
            theta = 90.0 degrees
            t = 42430.8 seconds after the module's landing sequence was initiated

    The spherical coordinates of our chosen landing site is:
        At the moment when we photographed it:
            r = 6763993.6 meters
            phi = 47.5 degrees
            theta = 90.0 degrees
            t = 23667.7 seconds after the module's landing sequence was initiated

        After we took the last photograph of Buttercup:
            r = 6763993.6 meters
            phi = 48.4 degrees
            theta = 90.0 degrees
            t = 29979.1 seconds after the module's landing sequence was initiated

        When we're ready to descend:
            r = 6763993.6 meters
            phi = 49.2 degrees
            theta = 90.0 degrees
            t = 36290.5 seconds after the module's landing sequence was initiated

        By the time we've successfully landed:
            r = 6763993.6 meters
            phi = 50.0 degrees
            theta = 90.0 degrees
            t = 42430.8 seconds after the module's landing sequence was initiated
    

        LANDING:
    Video recording started.
    Spacecraft boosted with delta-v (-1402.58, -476.328, -0) m/s.
    Performed automatic orientation:
    Time: 36290.5 s
    Position: (4.42086e+06, -1.54493e+07, 0) m
    Velocity: (2805.16, 952.657, 0) m/s
    Spacecraft fell until time 39541.7 s.
    Performed automatic orientation:
    Time: 39541.7 s1
    Position: (9.59327e+06, -5.41865e+06, 0) m
    Velocity: (-634.211, 5314.76, 0) m/s
    The distance from the surface 0.90 hours after we initiated the landing is 4253835.8 m

    Landing module launched at time 39541.7 s with delta-v (708.343, 843.632, -0) m/s.
    Parachute area: 129.9 m^2
    Lander fell until time 40713.8 s.
    Performed automatic orientation:
    Time: 40713.8 s
    Position: (7.28895e+06, 2.41901e+06, 0) m
    Velocity: (-4649.47, 6617.37, 0) m/s
    The distance from the surface 1.23 hours after we initiated the landing is 915876.7 m

    Landing thruster properties:
    Force: 677.391 N
    Minimum activation height: 50 m
    Camera pointing towards direction of motion with polar angle offset 1.5708 rad and azimuthal angle offset 0 rad.
    Parachute with area 129.9 m^2 deployed at time 40713.8 s.
    Lander fell until time 42118.8 s.
    Performed automatic orientation:
    Time: 42118.8 s
    Position: (4.02122e+06, 5.45656e+06, 0) m
    Velocity: (-17.4786, 2.38537, 0) m/s
    The distance from the surface 1.62 hours after we initiated the landing is 14221.3 m

    Lander fell until time 45280.8 s.
    Performed automatic orientation:
    Time: 45280.8 s
    Position: (3.9734e+06, 5.47404e+06, 0) m
    Velocity: (-14.2775, 6.63725, 0) m/s
    The distance from the surface 2.50 hours after we initiated the landing is 107.1 m

    Landing engine with thrust 677.391 N activated at time 45280.8 s.
    Lander reached the surface at time 45612.3 s.
    Successfully landed on planet 3 at time 45612.3 s with velocity 0.31727 m/s. Well done!
    *** Achievement unlocked: Touchdown! ***
    Landing site coordinates recorded:
    theta = 90 deg
    phi = 48.0983 deg
    Lander rested on surface until time 45630.8 s.
    Performed automatic orientation:
    Time: 45630.8 s
    Position: (3.96896e+06, 5.47713e+06, 0) m
    Velocity: (-12.5132, 9.06756, 0) m/s

    We reached Buttercup's surface 2.59 hours after we initiated the landing!
    Our landing position coordinates is (3969.0, 5477.1, 0.0) km
    Our chosen landing site's current position (4237.9, 5271.8, 0.0) km
    Our chosen landing site's azimuthal position has changed to 51.2046 degrees
    We landed 338.4 km away from the planned landing site

    XML file landing.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Video with 1000 frames saved to landing.xml.
'''