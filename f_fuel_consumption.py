import ast2000tools.constants as const

def fuel_consumption(fuel_m, N_H2, N_box, time, spacecraft_m, mean_f, fuel_loss_s, delta_v):
    m_H2 = const.m_H2                                       # mass of a H2 molecule [kg]
    tot_init_m = spacecraft_m + fuel_m + m_H2*N_H2*N_box    # the rocket engine's initial mass with fuel included [kg]
    thrust_f = N_box*mean_f/time                            # the combustion chamber's total thrust force [N]
    a = thrust_f/tot_init_m                                 # the rocket's acceleration [m/s**2]
    delta_t = delta_v/a                                     # time spent accelerating the rocket [s]
    tot_fuel_loss = abs(delta_t*fuel_loss_s*N_box)          # total fuel loss [kg]
    return tot_fuel_loss