import numpy as np

def MaxwellBoltzmann_v(m, k, T, v):
    return (m/(2*np.pi*k*T))**(3/2)*np.exp(-(1/2)*(m*v**2/(k*T)))*4*np.pi*v**2