#EGEN KODE
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import scipy.integrate as sp

utils.check_for_newer_version()
system = SolarSystem(83856)
mission = SpaceMission(83856)

def MaxwellBoltzmann_v(m, k, T, v):
    return (m/(2*np.pi*k*T))**(3/2)*np.exp(-(1/2)*(m*v**2/(k*T)))*4*np.pi*v**2

def main():
    '''
    A. Investigating Boltzmann Statistics
    '''
    '''
    Task 1
    '''
    
    def f(x, μ, σ):
        
        ''' normal probability distribution function '''
        
        return 1/(np.sqrt(2*np.pi)*σ)*np.exp(-0.5*((x - μ)/σ)**2)
    
    def P(a, b, μ, σ):

        '''
        P(a=<x=<b) shows us the likelyhood of a gas particle residing 
        in an area with an x-coordinate between a and b. We find this
        as a function of x by integrating over our normal probability 
        distribution function f defined above
        '''

        probability = sp.quad(f, a, b, args = (μ, σ))[0]
        return probability
    
    '''
    we chose two random values for μ and σ to test our function
    '''
    
    σ = 1.0
    μ = 0.0
    for i in range(1, 4):
        s = float(P(- i*σ + μ, i*σ + μ, μ, σ))
        print(f'With lower and upper boundaries -+ {i}σ: {s:.3f}')
    
    '''
    TODO THEORY
    Our function f(x) is at its maximum value when x = μ, which in turn gives us
    e^(-0.5*((x - μ)/σ)^2) = e^0 = 1, so f = 1/(√(2*π)*σ). The function value at half
    maximum is therefore f = 1/2*(√(2*π)*σ). We can use this to find out what x needs 
    to be for f to return this value, as e^(-0.5*((x - μ)/σ)^2) needs to be equal to 0.5. 
    This gives us the equation 
     -0.5*((x - μ)/σ)^2 = ln(0.5)
          ((x - μ)/σ)^2 = -2*ln(2^(-1))
              (x - μ)/σ = +-√(2*ln(2))
                  x - μ = +-√(2*ln(2))*σ
    Since the Gaussian probability distribution is symmetric, and |x - μ| is the width of 
    the curve going from the mean to either of the x-coordinates of where f(x) = f(μ)/2, 
    then by the definition of Full Width at Half Maximum we get FWHM = 2*|x - μ| = 2*√(2*ln(2))*σ 
    '''
    
    '''
    Task 2
    '''
    
    N_H2 = 10**5                # number of H_2 molecules
    T = 3000                    # temperature [K]
    m_H2 = const.m_H2           # mass of a H2 molecule [kg]
    k = const.k_B               # Boltzmann constant [m^2*kg/s^2/K]
    
    σ = np.sqrt(k*T/m_H2)       # the standard deviation of our particle velocities
    
    vx_lim = 2.5*10**4
    vx = np.linspace(- vx_lim, vx_lim, N_H2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    ax1.plot(vx, f(vx, μ, σ), color = 'violet')
    ax1.set_xlabel(r'velocity ($v_x$) [$\frac{m}{s}$] ')
    ax1.set_ylabel('probability density')
    ax1.set_title('Gaussian velocity distribution for the\none-dimensional ' + r'$v_x$ component')
    
    a = 5*10**3
    b = 30*10**3
    
    probability = P(a, b, μ, σ)
    print(f'The probability of a particle having the velocity v_x = {vx_lim:.0f} m/s when its velocity can go from v_xmin = {a:.0f} m/s to v_xmax = {b:.0f} m/s is {probability:.5f}, which equals about {probability*100:.2f}% of the particles')
    amount = N_H2*probability
    print(f'Multplying this result with the amount of particles N = {N_H2:.0f} gives us {amount:.0f}, which is an estimate of the amount particles which have this velocity')
    
    '''
    TODO THEORY
    Now we need to use the specific Maxwell-Boltzmann distribution function
    for the absolute velocity, as this function is not Gaussian and therefore
    cannot be generalized in the same way as the function for the x-component
    of the velocity. This is because the Maxwell-Boltzmann distribution function
    for absolute velocity does not give us a symmetrical curve, and a Gaussian
    curve is always symmetrical
    '''
    
    v = np.linspace(0, 3*10**4, N_H2)

    ax2.plot(v, MaxwellBoltzmann_v(m_H2, k, T, v), color = 'slateblue')
    ax2.set_xlabel(r'velocity ($v$) [$\frac{m}{s}$]')
    ax2.set_ylabel('probability density')
    ax2.set_title('Maxwell-Boltzmann velocity distribution for\nthe absolute velocity ' + r'$v$')

    plt.show()
    fig.savefig(f'/Users/paljettrosa/Documents/AST2000/Gaussian&Maxwell-Boltzmann.pdf')
    
    '''
    TODO THEORY
    This plot is not in conflict with our plot of the x-component of the velocity
    because the particles' absolute velocity is not necessarily negative just
    because their velocities' x-component is negative, as they have y- and z-
    components as well, and the absolute velocity is determined by √(vx^2 + vy^2 + vz^2)
    '''

if __name__ == '__main__':
    main()

'''
RESULTS:
    TASK 1:
With lower and upper boundaries -+ 1σ: 0.683
With lower and upper boundaries -+ 2σ: 0.954
With lower and upper boundaries -+ 3σ: 0.997

    TASK 2:
The probability of a particle having the velocity v_x = 25000 m/s when its velocity can go from v_xmin = 5000 m/s to v_xmax = 30000 m/s is 0.07760, which equals about 7.76% of the particles
Multplying this result with the amount of particles N = 100000 gives us 7760, which is an estimate of the amount particles which have this velocity
'''