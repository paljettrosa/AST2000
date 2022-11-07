#EGEN KODE
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from part6AD import *

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [10978]) 

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

''' from part 6 '''

gamma = 1.4                         # the heat capacity ratio
k_B = const.k_B                     # Boltzmann constant [m^2*kg/s^2/K]
u = 1.661e-27                       # atomic unit of mass [kg]
m_CH4 = (12.0107 + 4*1.00784)*u     # mass of CH4 molecule [kg]
m_CO = (12.0107 + 15.999)*u         # mass of CO molecule [kg]
m_N2O = (2*14.0067 + 15.999)*u      # mass of N2O molecule [kg]

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
'''
Task 1
'''

'''
TODO THEORY

Since the atmosphere follows the same velocity as Buttercup, it only has an azimuthal angular component. This is easy
to find, since it's constant. The angular velocity is omega = 2*pi/T, where T denotes Buttercup's rotational period in seconds
This means that the atmosphere moves with the rotational velocity w = r*omega = 2*pi*r/T, where r is the distance from
Buttercup's surface. Since v_drag is the velocity of the spacecraft with respect to the atmosphere, we can find this by using
our known velocity v with respect to Buttercup, and subtract the newly found expression for the atmosphere's velocity with 
respect to Buttercup. In the expression for air resistance, we're interested in the absolute value of v_drag, and not the
direction. We therefore use abs_v_drag = |v - w|, where v and w are vectors.
'''

def air_resistance(ri, A, vi):
    w = ri*2*np.pi/T*np.array([0, 1, 0])
    v_drag = np.linalg.norm(vi - w)
    if abs(rho(ri)) >= rho0:
        Fd = 1/2*rho0*Cd*A*v_drag**2
    elif rho(ri) < 1e-5:
        Fd = 0
    else:
        Fd = 1/2*rho(ri)*Cd*A*v_drag**2
    return Fd

'''
Task 2
'''

'''
TODO THEORY

We know that our spacecraft's velocity with respect to the planet can be written as v = vrad*erad + vtheta*etheta, and that
as we fall, the gravitational pull from our planet (which points radially inwards), as well as the air resistance, are forces
that work on us. The air resistance always works in the opposite direction of the movement, so it will work radially outwards.
As we fall, the gravitational force from Buttercup will drag us with it along its rotational direction, and shortly after we've
entered the atmosphere we'll have a rotational velocity determined by the distance r from Buttercup's surface. Our tangential
velocity vtheta will therefore become the same as w, which means we have zero tangential velocity with respect to the
atmosphere.

Since we no longer have a tangential velocity component with respect to the atmosphere after a short while, we only
have a radial velocity component. As we know, we reach a terminal velocity after falling for a while. This terminal velocity 
is constant for the rest of the fall, and since the tangential velocity tends to zero, this means that the radial velocity
component will stabilize on a constant velocity as we reach the terminal velocity.

When our velocity is constant, the net force on the spacecraft has to be zero. The gravitational force Fg from the planet is
given by Fg = - G*M/r^2*erad, where G is the gravitational constant and M is the mass of Buttercup. The air resistance is given
by Fd = 1/2*rho*Cd*A*v_drag^2*erad, and the sum of these two forces will be zero after we reack the terminal velocity vrad.
Since we know that v_drag = |v - w| = |vrad*(-erad) - 2*pi*r/T*etheta| = sqrt(vrad^2 + (2*pi*r/T)^2), we can use this together
with the fact that Fd + Fg = 0 to find the constant radial velocity vrad. We use h instead of r in the expression for w, to
make sure we don't confuse this r for altitude with the r used in the expression for Fg, which denotes the distance between the
spacecraft's center of mass and Buttercup's center of mass:

                  Fd + Fg = 0
    1/2*rho*Cd*A*v_drag^2 = G*m*M/r^2,                  m denotes the mass of the lander, M denotes the mass of Buttercup
    vrad^2 + (2*pi*h/T)^2 = 2*G*m*M/(r^2*rho*Cd*A)
                     vrad = sqrt(2*G*m*M/(r^2*rho*Cd*A) - (2*pi*h/T)^2)

We can rearrange this expression and solve it for A:

    vrad^2 + (2*pi*h/T)^2 = 2*G*m*M/(r^2*rho*Cd*A)
                        A = 2*G*m*M/(r^2*rho*Cd*(vrad^2 + (2*pi*h/T)^2))
                          = 2*G*m*M/(r^2*rho*Cd*(vrad^2 + w^2))

Close to the surface, the atmosphere moves (almost) with the same rotational velocity as the surface, so w tends to zero
(this is also apparent as h tends to zero when we approach the surface). Close to the surface, r tends to R, which is
Buttercup's radius, and rho tends to the atmospheric density by the surface provided to us by our research team. We can use
these known values to provide an estimate for A:

                        A = 2*G*m*M/(r^2*rho*Cd*(vrad^2 + w^2))
                        A = 2*G*m*M/(R^2*rho0*Cd*(3[m/s])^2)
                          ≈ (2 * 6.67*10^(-11)[N*kg^2/m^2] * 90.00[kg] * 5.22*10^24[kg])/((6763993.61[m])^2 * 1.17[kg/m^3] * 1 * (3[m/s])^2)
                          ≈ 129.90 m^2 
'''

vsafe = 3
A = 2*G*m_l*M/(R**2*rho0*Cd*vsafe**2)

'''
Task 3
'''

'''
#TODO THEORY

As we know, the safe velocity to land with is v_safe = vrad = 3m/s. We want to use a landing thruster to adjust our velocity
from the terminal velocity we reach from just falling, to the safe landing velocity. We still want our velocity to be constant,
so the sum of the air resistance Fd and the force Fl from the landing thruster (which will work against Fg to slow us down),
will have to be equal to Fg, so the net force on the spacecraft is zero. By the surface we have:

                    Fd + Fl = Fg
1/2*rho0*Cd*A*v_safe^2 + Fl = G*m*M/R^2

v_drag is the velocity of the spacecraft relative to the atmosphere, and we want this to be vsafe. Fd decreases as we
increase Fl, but Fg stays the same. We found an expression for Fg by the surface when there is no Fl, which is:

                         Fg = G*m*M/R^2
                            = 1/2*rho0*Cd*A*v_t^2 

where v_t is the reached terminal velocity. We can use this to derive a formula for the needed Fl:

1/2*rho0*Cd*A*v_safe^2 + Fl = 1/2*rho0*Cd*A*v_t^2
                         Fl = 1/2*rho0*Cd*A*(v_t^2 - v_safe^2)
                            = 1/2*rho0*A*(v_t^2 - v_safe^2)

where the last equality comes from the fact that we set Cd to be 1.
'''




'''
B. Simulating the Landing
'''

def simulate_landing(N, dt, t0, r0, v0, vsafe, m_sc, m_l, A_sc, A_l, A_p, vl, launch_pos, deploy_pos, thrust_pos): #TODO fjern unødvendig, burde vi heller ha launch_time og deploy_time?
    r = np.zeros((N, 3))
    v = np.zeros((N, 3))

    r[0] = r0
    v[0] = v0

    def gravity(m, ri):
        return np.array([- G*m*M/(R + ri)**2, 0, 0])
    
    def v_terminal(m, ri, A):
        if abs(rho(ri)) > rho0:
            return np.sqrt(2*abs(gravity(m, ri)[0])/(m*rho0*A*Cd))
        elif rho(ri) < 0:
            return 0
        else:
            return np.sqrt(2*abs(gravity(m, ri)[0])/(m*rho(ri)*A*Cd))

    def thrust(A, vt):                               #TODO riktig?
        return np.array([1/2*rho0*A*(vt**2 - vsafe**2), 0, 0])
    
    entered = False
    launched = False
    deployed = False
    thrusted = False
    landed = False
    tol = 5
    for i in range(N-1):
        if abs(v[i, 1]) <= 1e-3:
            Fd_direction = np.array([1, 0, 0])
        else:
            Fd_direction = - v[i]/np.linalg.norm(v[i])
        
        ''' checking if we've entered the atmosphere '''

        if rho(r[i, 0]) > 0 and entered == False:
            t_entered = i
            entered = True


        ''' checking if we're ready to launch the lander '''

        if abs(r[i, 0] - launch_pos) < tol and launched == False:
            t_launched = i
            v[i] = v[i] + vl
            launched = True
        

        ''' checking if we're ready to deploy our parachute '''

        if abs(r[i, 0] - deploy_pos) < tol*10 and deployed == False:
            t_deployed = i
            if rho(r[i, 0]) <= 0:
                print("You're deploying your parachute too high up in the atmosphere!")
                break
            else:
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

                if abs(v[i+1, 1]) <= 1e-4:
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
                a = (air_resistance(r[i, 0], A_l, v[i])*Fd_direction + gravity(m_l, r[i, 0]))/m_l
                v[i+1] = v[i] + a*dt/2
                r[i+1] = r[i] + v[i+1]*dt

                if abs(v[i+1, 1]) <= 1e-4:
                    Fd_direction = np.array([1, 0, 0])
                else:
                    Fd_direction = - v[i+1]/np.linalg.norm(v[i+1])
            
                a = (air_resistance(r[i+1, 0], A_l, v[i+1])*Fd_direction + gravity(m_l, r[i+1, 0]))/m_l
                v[i+1] = v[i+1] + a*dt/2


        ''' in case we've deployed our parachute '''

        if deployed == True and r[i, 0] > thrust_pos:
            if abs(v[i, 0]) >= v_terminal(m_l, r[i, 0], A_p):
                v[i, 0] = - v_terminal(m_l, r[i, 0], A_p)
                v[i+1] = v[i]
                r[i+1] = r[i] + v[i+1]*dt

            else:
                a = (air_resistance(r[i, 0], A_p, v[i])*Fd_direction + gravity(m_l, r[i, 0]))/m_l
                v[i+1] = v[i] + a*dt/2
                r[i+1] = r[i] + v[i+1]*dt

                if abs(v[i+1, 1]) <= 1e-4:
                    Fd_direction = np.array([1, 0, 0])
                else:
                    Fd_direction = - v[i+1]/np.linalg.norm(v[i+1])
            
                a = (air_resistance(r[i+1, 0], A_p, v[i+1])*Fd_direction + gravity(m_l, r[i+1, 0]))/m_l
                v[i+1] = v[i+1] + a*dt/2
        

        ''' in case we've started the landing thrusters '''

        if r[i, 0] <= thrust_pos:
            if thrusted == False:
                t_thrusted = i
                thrusted = True

            vt = v_terminal(m_l, r[i, 0], A_p)
            if abs(v[i, 0]) <= vsafe:
                a = 0
                v[i+1] = v[i]
            else:
                a = abs(thrust(A_p, vt)/m_l)
                v[i+1] = v[i] + a*dt/2
            r[i+1] = r[i] + v[i+1]*dt
            
            vt = v_terminal(m_l, r[i+1, 0], A_p)
            if abs(v[i+1, 0]) <= vsafe:
                a = 0
            else:
                a = abs(thrust(A_p, vt)/m_l)
                v[i+1] = v[i] + a*dt/2


            ''' checking if we've reached the surface '''

            if r[i+1, 0] < 1:
                if abs(v[i+1, 0]) > vsafe:
                    print('We crashlanded! Adjust the thrusters')
                else:
                    print("Woohoo! We've softly landed on Buttercup :)")
                    print(r[i+1, 0])
                    print(r[i, 1]*180/np.pi)
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
    

N = int(1e5)
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

#TODO fjern disse og kommenteringa
#t0 = 29979.104400000004
#r0 = np.array([1.49697178e07-R, 6.69766215e-02, 1.57079633e00])
#v0 = np.array([479.66108134, 4764.55428349/(1.49697178e07-R), 0])
#m_sc = 2869.051099400587

t, pos, vel, final_time, final_pos, final_vel, time_elapsed, t_entered, t_launched, t_deployed, t_thrusted = simulate_landing(N, dt, t0, r0, v0, vsafe, m_sc, m_l, A_sc, A_l, A, vl, launch_pos, deploy_pos, thrust_pos)

print('The spherical coordinates of our spacecraft is:')
print("  When we were about to descend:")
print(f'       r = {r0[0]:.1f} meters')
print(f'     phi = {r0[1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f'       t = {t0:.1f} seconds after the landing sequence was initiated\n')
print("  By the time we've successfully landed:")
print(f'       r = {final_pos[0]:.1f} meters')
print(f'     phi = {final_pos[1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f'       t = {final_time:.1f} seconds after the landing sequence was initiated\n')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t, pos[:, 0], color = 'palevioletred', label = 'radial trajectory', zorder = 1)
ax1.scatter(t[t_entered], pos[t_entered, 0], color = 'crimson', marker = 'o', s = 20, label = 'entering atmosphere')
ax1.scatter(t[t_launched], pos[t_launched, 0], color = 'crimson', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed], pos[t_deployed, 0], color = 'crimson', marker = '*', label = 'deploying parachute')
ax1.scatter(t[t_thrusted], pos[t_thrusted, 0], color = 'crimson', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'x-small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time (t) [s]')
ax1.set_ylabel('radial position (r) [m]')

ax2.plot(t, pos[:, 1]*180/np.pi, color = 'violet', label = 'azimuthal trajectory', zorder = 1)
ax2.scatter(t[t_entered], pos[t_entered, 1]*180/np.pi, color = 'darkmagenta', marker = 'o', s = 20, label = 'entering atmosphere')
ax2.scatter(t[t_launched], pos[t_launched, 1]*180/np.pi, color = 'darkmagenta', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed], pos[t_deployed, 1]*180/np.pi, color = 'darkmagenta', marker = '*', label = 'deploying parachute')
ax2.scatter(t[t_thrusted], pos[t_thrusted, 1]*180/np.pi, color = 'darkmagenta', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'x-small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal position $(\phi)$ [deg]')

fig.suptitle("Radial and azimuthal position during the entire landing")
plt.show()
fig.savefig(f'/Users/paljettrosa/Documents/AST2000/landing_trajectory.pdf')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t[t_entered-50:], pos[t_entered-50:, 0], color = 'hotpink', label = 'radial trajectory', zorder = 1)
ax1.scatter(t[t_entered], pos[t_entered, 0], color = 'deeppink', marker = 'o', s = 20, label = 'entering atmosphere')
ax1.scatter(t[t_launched], pos[t_launched, 0], color = 'deeppink', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed], pos[t_deployed, 0], color = 'deeppink', marker = '*', label = 'deploying parachute') 
ax1.scatter(t[t_thrusted], pos[t_thrusted, 0], color = 'deeppink', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'x-small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time (t) [s]')
ax1.set_ylabel('radial position (r) [m]')

ax2.plot(t[t_entered-50:], pos[t_entered-50:, 1]*180/np.pi, color = 'plum', label = 'azimuthal trajectory', zorder = 1)
ax2.scatter(t[t_entered], pos[t_entered, 1]*180/np.pi, color = 'mediumorchid', marker = 'o', s = 20, label = 'entering atmosphere')
ax2.scatter(t[t_launched], pos[t_launched, 1]*180/np.pi, color = 'mediumorchid', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed], pos[t_deployed, 1]*180/np.pi, color = 'mediumorchid', marker = '*', label = 'deploying parachute') 
ax2.scatter(t[t_thrusted], pos[t_thrusted, 1]*180/np.pi, color = 'mediumorchid', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'x-small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_xlabel('time (t) [s]')
ax2.set_ylabel(r'azimuthal position $(\phi)$ [deg]')

fig.suptitle("Close-up of the lander's trajectory after entering the atmosphere")
plt.show()
fig.savefig(f'/Users/paljettrosa/Documents/AST2000/landing_trajectory_closeups.pdf')



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t, vel[:, 0], color = 'palevioletred', label = 'radial velocity', zorder = 1)
ax1.scatter(t[t_entered], vel[t_entered, 0], color = 'crimson', marker = 'o', s = 20, label = 'entering atmosphere')
ax1.scatter(t[t_launched-1], vel[t_launched-1, 0], color = 'crimson', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed], vel[t_deployed, 0], color = 'crimson', marker = '*', label = 'deploying parachute')
ax1.scatter(t[t_thrusted], vel[t_thrusted, 0], color = 'crimson', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'x-small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time (t) [s]')
ax1.set_ylabel(r'radial velocity $(v_r)$ $[\frac{m}{s}]$')

ax2.plot(t, vel[:, 1]*180/np.pi, color = 'violet', label = 'azimuthal velocity', zorder = 1)
ax2.scatter(t[t_entered], vel[t_entered, 1]*180/np.pi, color = 'darkmagenta', marker = 'o', s = 20, label = 'entering atmosphere')
ax2.scatter(t[t_launched-1], vel[t_launched-1, 1]*180/np.pi, color = 'darkmagenta', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed], vel[t_deployed, 1]*180/np.pi, color = 'darkmagenta', marker = '*', label = 'deploying parachute')
ax2.scatter(t[t_thrusted], vel[t_thrusted, 1]*180/np.pi, color = 'darkmagenta', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'x-small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal velocity $(v_{\phi})$ $[\frac{deg}{s}]$')

fig.suptitle("Radial and azimuthal velocity during the entire landing")
plt.show()
fig.savefig(f'/Users/paljettrosa/Documents/AST2000/landing_velocity.pdf')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,)

ax1.plot(t[t_entered-50:], vel[t_entered-50:, 0], color = 'hotpink', label = 'radial velocity', zorder = 1)
ax1.scatter(t[t_entered], vel[t_entered, 0], color = 'deeppink', marker = 'o', s = 20, label = 'entering atmosphere')
ax1.scatter(t[t_launched-1], vel[t_launched-1, 0], color = 'deeppink', marker = 'd', s = 20, label = 'launching lander')
ax1.scatter(t[t_deployed], vel[t_deployed, 0], color = 'deeppink', marker = '*', label = 'deploying parachute') 
ax1.scatter(t[t_thrusted], vel[t_thrusted, 0], color = 'deeppink', marker = 'x', s = 20, label = 'starting thrusters')
ax1.legend(fontsize = 'x-small', facecolor = 'lightpink', edgecolor = 'black')
ax1.set_xlabel('time (t) [s]')
ax1.set_ylabel(r'radial velocity $(v_r)$ $[\frac{m}{s}]$')

ax2.plot(t[t_entered-50:], vel[t_entered-50:, 1]*180/np.pi, color = 'plum', label = 'azimuthal velocity', zorder = 1)
ax2.scatter(t[t_entered], vel[t_entered, 1]*180/np.pi, color = 'mediumorchid', marker = 'o', s = 20, label = 'entering atmosphere')
ax2.scatter(t[t_launched-1], vel[t_launched-1, 1]*180/np.pi, color = 'mediumorchid', marker = 'd', s = 20, label = 'launching lander')
ax2.scatter(t[t_deployed], vel[t_deployed, 1]*180/np.pi, color = 'mediumorchid', marker = '*', label = 'deploying parachute') 
ax2.scatter(t[t_thrusted], vel[t_thrusted, 1]*180/np.pi, color = 'mediumorchid', marker = 'x', s = 20, label = 'starting thrusters')
ax2.legend(fontsize = 'x-small', facecolor = 'thistle', edgecolor = 'black')
ax2.set_ylabel(r'azimuthal velocity $(v_{\phi})$ $[\frac{deg}{s}]$')

fig.suptitle("Close-up of the lander's velocity after entering the atmosphere")
plt.show()
fig.savefig(f'/Users/paljettrosa/Documents/AST2000/landing_velocity_closeups.pdf')





'''
C. Landing your Spacecraft
'''

''' we use the middle of the planet on the picture 'scouting8.png' as the landing site '''

print('The spherical coordinates of our chosen landing site is:')
print('  At the moment when we photographed it:')
print(f'       r = {coords[7, 0]:.1f} meters')
print(f'     phi = {coords[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f'       t = {coords[7, 3]:.1f} seconds after the landing sequence was initiated\n')
coords_afterlastpic = new_coords(coords, 0)
print("  After we took the last photograph of Buttercup:")
print(f'       r = {coords_afterlastpic[7, 0]:.1f} meters')
print(f'     phi = {coords_afterlastpic[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f'       t = {coords_afterlastpic[7, 3]:.1f} seconds after the landing sequence was initiated\n')
coords_aftercoast = new_coords(coords_afterlastpic, coast_time)
print("  When we're ready to descend:")
print(f'       r = {coords_aftercoast[7, 0]:.1f} meters')
print(f'     phi = {coords_aftercoast[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f'       t = {coords_aftercoast[7, 3]:.1f} seconds after the landing sequence was initiated\n')
coords_whenlanded = new_coords(coords_aftercoast, time_elapsed)
print("  By the time we've successfully landed:")
print(f'       r = {coords_whenlanded[7, 0]:.1f} meters')
print(f'     phi = {coords_whenlanded[7, 1]*180/np.pi:.1f} degrees')
print(f'   theta = 90.0 degrees')
print(f'       t = {coords_whenlanded[7, 3]:.1f} seconds after the landing sequence was initiated\n')                        

landing_site = np.array([coords_whenlanded[7, 0]*np.cos(coords_whenlanded[7, 1]), coords_whenlanded[7, 0]*np.sin(coords_whenlanded[7, 1]), 0])
final_pos_cartesian = np.array([final_pos[0]*np.cos(final_pos[1]), final_pos[0]*np.sin(final_pos[1]), 0])
distance_from_site = final_pos_cartesian - landing_site     # how far off we are from our chosen landing site
vboost = - distance_from_site/time_elapsed                  # how much we need to boost our rocket to land on the correct spot


''' landing '''

landing.start_video()
landing.boost(- velocity/3)                         #slowing down
landing.fall_until_time(t0 + t_launched*dt)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {time/(60*60):.1f} hours after we initiated the landing sequence is {r:.1f} m\n')

landing.launch_lander(vboost)
landing.adjust_parachute_area(A)
landing.fall_until_time(t0 + t_deployed*dt)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {time/(60*60):.1f} hours after we initiated the landing sequence is {r:.1f} m\n')

landing.deploy_parachute()
vt = np.sqrt(2*G*M/((R + 100)**2*rho(100)*A*Cd))
landing.adjust_landing_thruster(-1/2*rho0*A*(vt**2 - vsafe**2), 50)

landing.look_in_direction_of_motion(relative_polar_angle=np.pi/2, relative_azimuth_angle=0)
landing.fall_until_time(t0 + t_thrusted*dt)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {time/(60*60):.1f} hours after we initiated the landing sequence is {r:.1f} m\n')

landing.fall_until_time(final_time + 2850)
time, position, velocity = landing.orient()
r = np.linalg.norm(position) - R
print(f'The distance from the surface {time/(60*60):.1f} hours after we initiated the landing sequence is {r:.1f} m\n')

landing.activate_landing_thruster()
landing.fall_until_time(final_time + 3200)
time, position, velocity = landing.orient()

if landing.reached_surface == True:
    site_pos = new_coords(coords, time - t0)
    new_landing_site_coordinates = np.array([site_pos[7, 0]*np.cos(site_pos[7, 1]), site_pos[7, 0]*np.sin(site_pos[7, 1]), 0])
    distance_from_site = np.linalg.norm(position - new_landing_site_coordinates)
    print('\n')
    print(f"We reached Buttercup's surface {time/(60*60):.1f} hours after we initiated the landing sequence!")
    print(f"Our landing position coordinates is ({position[0]*1e-3:.1f}, {position[1]*1e-3:.1f}, 0.0) km")  
    print(f"Our chosen landing site's current position ({new_landing_site_coordinates[0]*1e-3:.1f}, {new_landing_site_coordinates[1]*1e-3:.1f}, 0.0) km")
    print(f"We landed {distance_from_site*1e-3:.1f} km away from the planned landing site\n")

landing.finish_video(r'/Users/paljettrosa/Documents/AST2000/MCast/data/landing.xml')




'''
RESULTS:
        FROM SIUMULATION:
    Woohoo! We've softly landed on Buttercup :)
    

        COORDINATES:
    The spherical coordinates of our spacecraft during the simulated landing is:
        When we were about to descend:
            r = 8205724.2 meters
          phi = 3.8 degrees
        theta = 90.0 degrees
            t = 29979.1 seconds after the landing sequence was initiated

        By the time we've successfully landed:
            r = 1.0 meters
          phi = 63.7 degrees
        theta = 90.0 degrees
            t = 36216.4 seconds after the landing sequence was initiated

    The spherical coordinates of our chosen landing site is:
        At the moment when we photographed it:
            r = 6763993.6 meters
          phi = 47.5 degrees
        theta = 90.0 degrees
            t = 23667.7 seconds after the landing sequence was initiated

        When we're ready to descend:
            r = 6763993.6 meters
          phi = 48.4 degrees
        theta = 90.0 degrees
            t = 29979.1 seconds after the landing sequence was initiated

        By the time we've successfully landed:
            r = 6763993.6 meters
          phi = 49.2 degrees
        theta = 90.0 degrees
            t = 36216.4 seconds after the landing sequence was initiated
    
        LANDING:
    Video recording started.
    Spacecraft boosted with delta-v (-1402.58, -476.328, -0) m/s.
    Spacecraft fell until time 39541.8 s.
    Performed automatic orientation:
    Time: 39541.8 s
    Position: (9.59321e+06, -5.41811e+06, 0) m
    Velocity: (-634.46, 5314.9, 0) m/s
    The distance from the surface 11.0 hours after we initiated the landing sequence is 4253519.2 m

    Landing module launched at time 39541.8 s with delta-v (708.261, 843.537, -0) m/s.
    Parachute area: 129.9 m^2
    Lander fell until time 40713.4 s.
    Performed automatic orientation:
    Time: 40713.4 s
    Position: (7.29061e+06, 2.41621e+06, 0) m
    Velocity: (-4647.44, 6618.08, 0) m/s
    The distance from the surface 11.3 hours after we initiated the landing sequence is 916567.5 m

    Landing thruster properties:
        Force: 677.39 N
        Minimum activation height: 50 m
    Camera pointing towards direction of motion with polar angle offset 1.5708 rad and azimuthal angle offset 0 rad.
    Parachute with area 129.9 m^2 deployed at time 40713.4 s.
    Lander fell until time 42119.5 s.
    Performed automatic orientation:
    Time: 42119.5 s
    Position: (4.02431e+06, 5.45426e+06, 0) m
    Velocity: (-17.4703, 2.40477, 0) m/s
    The distance from the surface 11.7 hours after we initiated the landing sequence is 14210.9 m

    Lander fell until time 45281.5 s.
    Performed automatic orientation:
    Time: 45281.5 s
    Position: (3.97651e+06, 5.47178e+06, 0) m
    Velocity: (-14.2734, 6.6458, 0) m/s
    The distance from the surface 12.6 hours after we initiated the landing sequence is 103.5 m

    Landing engine with thrust 677.39 N activated at time 45281.5 s.
    Lander reached the surface at time 45601.4 s.
    Successfully landed on planet 3 at time 45601.4 s with velocity 0.31729 m/s. Well done!
    *** Achievement unlocked: Touchdown! ***
    Landing site coordinates recorded:
        theta = 90 deg
        phi = 48.0656 deg
    Lander rested on surface until time 45631.5 s.
    Performed automatic orientation:
    Time: 45631.5 s
    Position: (3.97208e+06, 5.47487e+06, 0) m
    Velocity: (-12.508, 9.07468, 0) m/s

    We reached Buttercup's surface 12.7 hours after we initiated the landing sequence!
    Our landing position coordinates is (3972.1, 5474.9, 0.0) km
    Our chosen landing site's current position (4237.9, 5271.8, 0.0) km
    We landed 334.5 km away from the planned landing site

    XML file /Users/paljettrosa/Documents/AST2000/MCast/data/landing.xml was saved in XMLs/.
    It can be viewed in MCAst.
    Video with 1000 frames saved to /Users/paljettrosa/Documents/AST2000/MCast/data/landing.xml.
'''