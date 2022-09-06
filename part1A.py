import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import ast2000tools.constants as const

'''
1. A First Meeting with the Gaussian Distribution
'''
#1.1)

def f(x, my, sigma):
    
    # normal probability distribution function ???
    
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((x - my)/sigma)**2)

def P(a, b, my, sigma):
    
    # probability density function ???

    probability = sp.quad(f, a, b, args = (my, sigma))[0]
    return probability

#1.2)

'''
P(a =< x =< b) shows us the probability of a gas particle residing 
in an area with an x-coordinate between a and b (???) f er probability density function
'''

#1.3)

'''
we chose two random values for my and sigma to test our function

'''
sigma = 1.0
my = 0.0
sigma_list = [sigma, 2*sigma, 3*sigma]
for i in range(len(sigma_list)):
    s = float(P(- sigma_list[i] + my, sigma_list[i] + my, my, sigma))
    #print(s)
    print(f'{s:.3f}')

#1.4)

'''
f(x) is at it's maximum value when x = my, which in turn gives us
sp.exp(-0.5*((x - my)/sigma)**2) = sp.exp(0) = 1, so f = 1/(sp.sqrt(2*sp.pi)*sigma)
the function value at half maximum is therefore f = 1/2*(sp.sqrt(2*sp.pi)*sigma)
we can use this to find out what x needs to be for f to return this value
as sp.exp(-0.5*((x - my)/sigma)**2) needs to be equal to 0.5. this gives us
the equation
-0.5*((x - my)/sigma)**2 = ln(0.5)
((x - my)/sigma)**2 = -2*ln(2**(-1))
(x - my)/sigma = +-sqrt(2*ln(2))
x = +-sqrt(2*ln(2))*sigma + my
hvordan bli kvitt my ???
since this is the positive and the negative x-coordinate that gives us the half
maximum, the definition of FWHM tells us that FWHM = |x| = 2*sqrt(2*ln(2))*sigma (?)
'''

'''
2. The Maxwell-Boltzmann Distribution
'''
#2.1)

N = 10**5                   # number of H_2 molecules
T = 3000                    # temperature [K]
m = const.m_H2              # mass of a H2 molecule [kg]
k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]

sigma = np.sqrt(k*T/m)      # the standard deviation of our particle velocities

vx_lim = 2.5*10**4
vx = np.linspace(- vx_lim, vx_lim, N)

plt.subplot(2, 1, 1)
plt.plot(vx, f(vx, my, sigma))

#2.2)

a = 5*10**3
b = 30*10**3
my = 0.0

prob = P(a, b, my, sigma)
print(prob)

'''
0.07759630146176734 is the probability of a particle having this velocity
this equals to about 7.76% of the particles
'''

amount = N*prob
print(amount)

'''
7759.630146176734 is the approximate amount of particles with a velocity
within this range
'''

#2.3)

'''
here we need to use the specific Maxwell - Boltzmann distribution function
for the absolute velocity, as this function is not Gaussian and therefore
cannot be generalized in the same way as the function for the x-component
of the velocity. this is because the axwell - Boltzmann distribution function
for absolute velocity does not give us a symmetrical curve, and a Gaussian
curve is always symmetrical
'''
def MaxwellBoltzmann_v(m, k, T, v):
    return (m/(2*np.pi*k*T))**(3/2)*np.exp(-(1/2)*(m*v**2/(k*T)))*4*np.pi*v**2

v = np.linspace(0, 3*10**4, N)

plt.subplot(2, 1, 2)
plt.plot(v, MaxwellBoltzmann_v(m, k, T, v))

'''
this plot is not in conflict with our plot of the x-component of the velocity
because the particles' absolute velocity is not necessarily negative just
because their velocities' x-component is negative, as they have y- and z-
components as well, and the absolute velocity is determined by the square-root
og vx^2 + vy^2 + vz^2
'''

'''
3. Elementary Statistical Physics
'''

#3.1)

#FIKS 3.3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!