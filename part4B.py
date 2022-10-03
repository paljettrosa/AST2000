#EGEN KODE
from libraries import *
from PIL import Image

'''
B. Image Analysis
'''
def find_phi0(pic_name, width = 640, length = 480, theta0 = np.pi/2):
    pic = Image.open(pic_name)
    pixels = np.array(pic)
    center_pixels = pixels[(int(length/2)-2):(int(length/2)+2), (int(width/2)-2):(int(width/2)+2)]
    for i in range(360):
        ref_pic = Image.open(f'phi_{i}.0_deg.png')
        ref_pixels = np.array(ref_pic)
        center_ref_pixels = ref_pixels[(int(length/2)-2):(int(length/2)+2), (int(width/2)-2):(int(width/2)+2)]
        if np.all(center_pixels == center_ref_pixels):
            phi0 = i
            print(f"The angle that the spacecraft was facing when {pic_name} was taken was at phi = {phi0} degrees")
            break
    return phi0
    
phi0 = find_phi0('phi_42.0_deg.png')

'''
C. Doppler Shift Analysis
'''
'''
Task 1
'''
c = const.c                                                 # the speed of light [m/s]
lambda0 = mission.reference_wavelength                      # the wavelength of the spectral line of H-alpha at rest [nm]
phi = np.array(mission.star_direction_angles)               # the azimuthal angles of the stars relative to the solar system x-axis in degrees

dlambda_sun = np.array(mission.star_doppler_shifts_at_sun)  # the Doppler shift of the stars relative to our sun [nm]
vr_sun = c*dlambda_sun/lambda0                              # radial velocity of our sun relative to the stars [m/s]

dlambda_sc = np.array([0.0, 0.0])                           # the Doppler shift of the stars relative to the spacecraft [nm]
vr_sc = c*dlambda_sc/lambda0                                # radial velocity of our spacecraft relative to the stars [m/s]

'''
since our spacecraft's spectrograph measures dlambda = 0 for both radial 
velocities, our spacecraft does not have a radial velocity component in
respect to either of the reference stars
'''

def phi_vel(vr_sun, vr_sc, phi):
    phi = phi*np.pi/180                     # converting to radians
    
    '''defining a new coordinate system using the method introduced in the relevant mathematics'''
    
    u = np.zeros((2, 2))
    for i in range(2):
        u[i] = np.array([np.cos(phi[i]), np.sin(phi[i])])                               # defining the new coordinate vectors
    M = 1/np.sin(phi[1] - phi[0])*np.array([[np.sin(phi[1]), -np.sin(phi[0])],          # the change of coordinates matrix 
                                            [-np.cos(phi[1]), np.cos(phi[0])]])
    
    vphi = vr_sun - vr_sc                   # the velocity of the spacecraft relative to our sun in (phi1, phi2)-coordinates [m/s]
    vxy = np.matmul(M, vphi)                # the velocity of the spacecraft relative to our sun in (x, y)-coordinates [m/s]
    
    #ER IKKE VPHI EGT BARE RADIELL HASTIGHET TIL ROMSKIPET RELATIVT TIL SOLA?????
    #VPHI ER VR, OG VI MÅ GANGE VR MED U-VEKTOR??????
    return vphi, vxy

