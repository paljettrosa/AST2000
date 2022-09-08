def fuel_consumption(N_box, thrust_f, initial_m, fuel_loss_s, delta_v):
    '''
    we assume that the amount of fuel that the rocket loses during the speed boost
    is so minimal that we can define the rocket's acceleration as the total thrust force
    divided by it's total mass before the boost
    '''
    a = thrust_f/initial_m                                  # the rocket's acceleration [m/s**2]
    delta_t = delta_v/a                                     # time spent accelerating the rocket [s]
    tot_fuel_loss = abs(delta_t*fuel_loss_s*N_box)          # total fuel loss [kg]
    return tot_fuel_loss
