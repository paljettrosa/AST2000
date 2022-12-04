#EGEN KODE
#KANDIDATER 15361 & 15384
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
plt.rcParams.update({'font.size': 12})

def MaxwellBoltzmann_v(m, k, T, v):

    '''
    Maxwell-Boltzmann Distribution for absolute velocity
    '''

    return (m/(2*np.pi*k*T))**(3/2)*np.exp(-(1/2)*(m*v**2/(k*T)))*4*np.pi*v**2

def main():
    '''
    A. Investigating Boltzmann Statistics
    '''
    '''
    Task 1
    '''
    
    def f(x, μ, σ):
        
        ''' 
        normal probability distribution function 
        '''
        
        return 1/(np.sqrt(2*np.pi)*σ)*np.exp(-0.5*((x - μ)/σ)**2)
    
    def P(a, b, μ, σ):

        '''
        function for integrating probability ditribution
        '''

        probability = sp.quad(f, a, b, args = (μ, σ))[0]
        return probability
    

    ''' we choose two random values for μ and σ to test our function '''
    
    σ = 1.0
    μ = 0.0
    for i in range(1, 4):
        s = float(P(- i*σ + μ, i*σ + μ, μ, σ))
        print(f'With lower and upper boundaries -+ {i}σ: {s:.3f}')

    
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
    print(f"The probability of a particle's velocity residing withing the interval [{a:.0f} m/s, {b:.0f} m/s] when its velocity can reside within the interval [-{vx_lim:.0f} m/s, {vx_lim:.0f} m/s] is {probability:.5f}, which equals about {probability*100:.2f}% of the particles")
    amount = N_H2*probability
    print(f'Multiplying this result with the amount of particles N = {N_H2:.0f} gives us {amount:.0f}, which is an estimate of the amount particles that have this velocity')
    
    v = np.linspace(0, 3*10**4, N_H2)

    ax2.plot(v, MaxwellBoltzmann_v(m_H2, k, T, v), color = 'slateblue')
    ax2.set_xlabel(r'velocity ($v$) [$\frac{m}{s}$]')
    ax2.set_ylabel('probability density')
    ax2.set_title('Maxwell-Boltzmann velocity distribution for\nthe absolute velocity ' + r'$v$')

    plt.show()
    fig.savefig('Gaussian&Maxwell-Boltzmann.pdf')
    

if __name__ == '__main__':
    main()

'''
RESULTS:
    TASK 1:
With lower and upper boundaries -+ 1σ: 0.683
With lower and upper boundaries -+ 2σ: 0.954
With lower and upper boundaries -+ 3σ: 0.997

    TASK 2:
The probability of a particle's velocity residing withing the interval [5000 m/s, 30000 m/s] when its velocity 
can reside within the interval [-25000 m/s, 25000 m/s] is 0.07760, which equals about 7.76% of the particles
Multiplying this result with the amount of particles N = 100000 gives us 7760, which is an estimate of the amount 
particles that have this velocity
'''