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
def dms_to_degrees(degrees, arcmin, arcsec):
     if degrees < 0.0:
          return degrees - (arcmin/60.0) - (arcsec/3600.0)
     else:
          return degrees + (arcmin/60.0) + (arcsec/3600.0)
#ra = dms_to_degrees()/15
e = np.radians(23.43713)
ra, dec = '20 28 07.25', '-14 56 18.2'
ra = np.fromstring(ra,sep=' ')
dec = np.fromstring(dec,sep = ' ')
ra = np.radians(dms_to_degrees(ra[0],ra[1],ra[2])*15.)
dec = np.radians(dms_to_degrees(dec[0],dec[1],dec[2]))

v_equ = vector(cos(ra)*cos(dec),sin(ra)*cos(dec),sin(dec))
v_ecl = vector(cos(ra)*cos(dec),sin(ra)*cos(dec)*cos(e)+sin(dec)*sin(e),-sin(ra)*cos(dec)*sin(e)+sin(dec)*cos(e))

lat = np.arcsin(v_ecl.z)
lon = np.arcsin(v_ecl.y/cos(lat))

r_prime = rotate(v_equ, radians(e),axis = vector(1,0,0))

ra, dec, lon, lat = degrees(ra), degrees(dec), degrees(lon), degrees(lat)

"""
x_axis = arrow(pos=(0,0,0),axis = (2,0,0),shaftwidth =0.01, color = color.blue)
y_axis = arrow(pos=(0,0,0),axis = (0,2,0),shaftwidth =0.01, color = color.green)
z_axis = arrow(pos=(0,0,0),axis = (0,0,2),shaftwidth =0.01, color = color.red)
earth = sphere(pos = (0,0,0), radius = 2, color = color.green, opacity = 0.1)
horizon = cylinder(pos = (0,0,0), axis = (0,0,0.01), radius = 2, opacity = 0.5, color = color.yellow)
ecliptic = cylinder(pos = (0,0,0), axis = rotate(horizon.axis, radians(e),axis = vector(1,0,0)), radius = 2, opacity = 0.5)
# 2017-Jul-14 04:00  20 28 14.82 -14 49 14.2
label(pos = (1,0,0), text = 'x axis', height = 6, box = False, opacity = 0)
label(pos = (0,1,0), text = 'y axis', height = 6, box = False, opacity = 0)
label(pos = (0,0,1), text = 'z axis', height = 6, box = False, opacity = 0)
label(pos = (2,0,0), text = 'vernal equinox', height = 6, box = False, opacity = 0)
label(pos = (-2,0,2), text = 'Yellow: equatorial, White: ecliptical', box = False, opacity = 0)


arrow(pos = (0,0,0), axis = r, color = color.yellow)
arrow(pos = (0,0,0), axis = r_prime )
print r, r_prime
"""
