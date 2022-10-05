#EGEN KODE
import numpy as np
from PIL import Image
import imageio as ig
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

utils.check_for_newer_version()
seed = utils.get_seed('somiamc')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planets = np.array([['Doofenshmirtz', 'black'], ['Blossom', 'crimson'], 
                    ['Bubbles', 'skyblue'], ['Buttercup', 'olivedrab'], 
                    ['Flora', 'pink'], ['Stella', 'gold'], ['Aisha', 'darkorchid']])

'''
A. Generating Reference Pictures
'''
'''
Task 2
'''
img = Image.open('sample0000.png')
pixels = np.array(img)
width = len(pixels[0, :])
length = len(pixels[:, 0])
size = width*length
print(f"There are {width} pixels in the horizontal direction, {length} pixels in the vertical direction, and the picture consists of a total of {size} pixels")

def xy_limits(a_theta, a_phi):
    xmax = (2*np.sin(a_phi/2))/(1 + np.cos(a_phi/2))
    xmin = -(2*np.sin(a_phi/2))/(1 + np.cos(a_phi/2))
    ymax = -(2*np.sin(a_theta/2))/(1 + np.cos(a_theta/2))
    ymin = (2*np.sin(a_theta/2))/(1 + np.cos(a_theta/2))
    return xmin, xmax, ymin, ymax

def stereographic_projection(X, Y, theta0 = np.pi/2, phi0 = 0):
    rho = np.sqrt(X**2 + Y**2)             # the absolute distance from the origin
    beta = 2*np.arctan(rho/2) 
    with np.errstate(divide = "ignore", invalid = "ignore"):   
        theta = theta0 - np.arcsin(np.cos(beta)*np.cos(theta0) + Y/rho*np.sin(beta)*np.sin(theta0))
        phi = phi0 + np.arctan(X*np.sin(beta)/(rho*np.sin(theta0)*np.cos(beta) - Y*np.cos(theta0)*np.sin(beta)))
    return theta, phi

a_theta = 70*np.pi/180
a_phi = 70*np.pi/180

xmin, xmax, ymin, ymax = xy_limits(a_theta, a_phi)

x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, length)
X, Y = np.meshgrid(x, y)

theta0 = np.pi/2
phi0 = 0

theta, phi = stereographic_projection(X, Y)

def generate_picture(theta, phi, width = 640, length = 480, theta0 = np.pi/2, phi0 = 0):
    sky_sphere = np.load('himmelkule.npy')
    pxl_idx = np.zeros((length, width))
    pixels = np.zeros((length, width, 3), dtype = "uint8")
    for i in range(length):
        for j in range(width):
            pxl_idx[i, j] = mission.get_sky_image_pixel(theta[i, j], phi[i, j])
            for k in range(3):
                pixels[i, j, k] = sky_sphere[int(pxl_idx[i, j]), k+2]
    picture = Image.fromarray(pixels)
    return picture

picture = generate_picture(theta, phi)
picture.save('sample0000_recreated.png')

'''
Task 3
'''

def generate_picture_360(X, Y, phi_array, width = 640, length = 480, theta0 = np.pi/2):
    sky_sphere = np.load('himmelkule.npy')
    pxl_idx = np.zeros((length, width))
    pixels = np.zeros((length, width, 3), dtype = "uint8")
    gif_array = []
    for phi0 in phi_array:
        theta, phi = stereographic_projection(X, Y, theta0, phi0*np.pi/180)
        for i in range(length):
            for j in range(width):
                pxl_idx[i, j] = mission.get_sky_image_pixel(theta[i, j], phi[i, j])
                for k in range(3):
                    pixels[i, j, k] = sky_sphere[int(pxl_idx[i, j]), k+2]
        picture = Image.fromarray(pixels)
        picture.save(f'phi_{phi0}_deg.png')
        gif_array.append(ig.imread(f'phi_{phi0}_deg.png'))
    ig.mimsave('spacecraft_position.gif', gif_array) 

phi_array = np.linspace(0, 359, 3600)
generate_picture_360(X, Y, phi_array)

'''
There are 640 pixels in the horizontal direction, 480 pixels in the vertical 
direction, and the picture consists of a total of 307200 pixels
'''
