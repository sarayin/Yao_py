import numpy as np
from visual import *
cos, sin = np.cos, np.sin
# =======================
# Problem 1
# =======================
"""
lat = np.radians(51.1789)
dec = np.radians(-23.4373)
HA = np.arccos(-sin(lat)*sin(dec)/(cos(lat)*cos(dec)))
A = np.arcsin(-cos(dec)*sin(HA))
"""

# ======================
# Problem 2
# ======================
#The function that will convert the Ra and Dec from JPL horizons to decimal degrees
def dms_to_degrees(degrees, arcmin, arcsec):
     if degrees < 0.0:
          return degrees - (arcmin/60.0) - (arcsec/3600.0)
     else:
          return degrees + (arcmin/60.0) + (arcsec/3600.0)

e = np.radians(23.43713) #tilt of the earth in 2017
ra, dec = '20 28 07.25', '-14 56 18.2' #Ra, Dec found on JPL horizons
#Convert Ra and Dec to decimal degrees/radians
ra = np.fromstring(ra,sep=' ')
dec = np.fromstring(dec,sep = ' ')
ra = np.radians(dms_to_degrees(ra[0],ra[1],ra[2])*15.)
dec = np.radians(dms_to_degrees(dec[0],dec[1],dec[2]))
#Therefore, the unit vector in the equatorial coordinate system is:
v_equ = vector(cos(ra)*cos(dec),sin(ra)*cos(dec),sin(dec))

#Using the multiplication of vectors, we can convert the vector in the equtorial system to the ecliptic system
trans_matrix = np.array([[1,0,0],[0,cos(e),sin(e)],[0,-sin(e),cos(e)]])
v_ecl = vector(np.matmul(trans_matrix, v_equ))
#We can also use the rotate function in vPython to generate a unit vector in the ecliptic coordinate system
r_prime = rotate(v_equ,-e,axis = vector(1,0,0)) #This vector should be the same as v_ecl

#Then we can use the vector to find the ecliptic longitude and latitude of our asteroid
lat = np.arcsin(v_ecl.z)
lon = np.arcsin(v_ecl.y/cos(lat))

ra, dec, lon, lat = degrees(ra), degrees(dec), degrees(lon), degrees(lat)


x_axis = arrow(pos=(0,0,0),axis = (2,0,0),shaftwidth =0.01, color = color.blue)
y_axis = arrow(pos=(0,0,0),axis = (0,2,0),shaftwidth =0.01, color = color.green)
z_axis = arrow(pos=(0,0,0),axis = (0,0,2),shaftwidth =0.01, color = color.red)
earth = sphere(pos = (0,0,0), radius = 2, color = color.green, opacity = 0.1)
horizon = cylinder(pos = (0,0,0), axis = (0,0,0.01), radius = 2, opacity = 0.5, color = color.yellow)
ecliptic = cylinder(pos = (0,0,0), axis = rotate(horizon.axis, e,axis = vector(1,0,0)), radius = 2, opacity = 0.5)
# 2017-Jul-14 04:00  20 28 14.82 -14 49 14.2
label(pos = (1,0,0), text = 'x axis', height = 6, box = False, opacity = 0)
label(pos = (0,1,0), text = 'y axis', height = 6, box = False, opacity = 0)
label(pos = (0,0,1), text = 'z axis', height = 6, box = False, opacity = 0)
label(pos = (2,0,0), text = 'vernal equinox', height = 6, box = False, opacity = 0)
label(pos = (-2,0,2), text = 'Yellow: equatorial, Cyan: ecliptical', box = False, opacity = 0)

arrow(pos = (0,0,0), axis = v_equ, color = color.yellow)
arrow(pos = (0,0,0), axis = v_ecl, color = color.cyan)
