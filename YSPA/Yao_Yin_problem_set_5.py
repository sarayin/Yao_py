import numpy as np
from yao_master_script import *
from visual import *
from math import *

sin, cos = np.sin, np.cos
"""
#1
def ra_dec_to_vec(ra,dec):
    #ra, dec = '20 28 07.25', '-14 56 18.2' #Ra, Dec found on JPL horizons
    #Convert Ra and Dec to decimal degrees/radians
    ra = np.fromstring(ra,sep=' ')
    dec = np.fromstring(dec,sep = ' ')
    ra = np.radians(dms_to_degrees(ra[0],ra[1],ra[2])*15.)
    dec = np.radians(dms_to_degrees(dec[0],dec[1],dec[2]))
    #Therefore, the unit vector in the equatorial coordinate system is:
    vec = vector(cos(ra)*cos(dec),sin(ra)*cos(dec),sin(dec))
    return vec

Altair_vec = ra_dec_to_vec('19 51 0','8 52 0')
Vega_vec = ra_dec_to_vec('18 37 0','38 47 0')
theta = np.arccos(np.dot(Altair_vec,Vega_vec)/(mag(Altair_vec)*mag(Vega_vec)))

#2a
def lon_lat_to_vec(lon,lat):
    lon, lat = np.radians(lon),np.radians(lat)
    vec = vector(cos(lon)*cos(lat),sin(lon)*cos(lat),sin(lat))
    return vec

NewHaven_vec = lon_lat_to_vec(-72.93,41.31)*6371
Boulder_vec = lon_lat_to_vec(-105.27,40.02)*6371
angle = np.arccos(np.dot(NewHaven_vec,Boulder_vec) / (mag(NewHaven_vec)*mag(Boulder_vec)))
dist = 40075 * (angle/(2*pi)) #Earth's diameter = 40075 km

#2b
diff_vec = Boulder_vec-NewHaven_vec #Eath radius = 6371 km
new_dist = mag(diff_vec)

#2d
NAsteroid_vec = NewHaven_vec/mag(NewHaven_vec) * (4.488e+7+6371) #0.3AU = 4.488e+7 km, Vector that points from New Haven to the asteroid
BAsteroid_vec = NAsteroid_vec-diff_vec

#2e
parallex_angle = np.arccos(np.dot(NAsteroid_vec,BAsteroid_vec)/(mag(NAsteroid_vec)*mag(BAsteroid_vec)))
"""


Juliet = sphere(pos = (0.5,0,0), axis = (1,0,0), radius = 0.5,color = color.red)
Romeo = sphere(pos = (0,0,0.1), axis = (1,0,0), radius = 0.5,color = color.blue)
J_trail = curve(color = color.red)
R_trail = curve(color = color.blue)
t = [0]
R = [Romeo.pos.x]
J = [Juliet.pos.x]

i = 0
delta_t = 0.1
while i<1000:
    rate(100)
    vel_x = 0.05
    drdt = Juliet.pos.x
    djdt = -Romeo.pos.x
    Juliet.pos.x += djdt * delta_t
    Romeo.pos.x += drdt * delta_t
    R.append(Romeo.pos.x)
    J.append(Juliet.pos.x)
    t.append(t[-1]+delta_t)
    i+=1

fig,ax = plt.subplots()
ax.plot(t,J,'r-')
ax.plot(t,R,'b-')
plt.show(fig)
