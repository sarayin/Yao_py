from visual import *
import numpy as np
from math import *

k = 2*pi/365. #modified day per day
print k
e = np.radians(23.43713)
xaixis = arrow(pos=(0,0,0),axis=(1,0,0),shaftwidth=0.03, length = 10, color = color.blue)
yaixis = arrow(pos=(0,0,0),axis=(0,1,0),shaftwidth=0.03, length = 10, color = color.green)
zaixis = arrow(pos=(0,0,0),axis=(0,0,1),shaftwidth=0.03, length = 10,color = color.orange)
ecliptic = cylinder(pos=(0,0,0), axis = (0,0,1), radius = 10, length =0.01,opacity=0.1)
sun = sphere(pos = (0,0,0), radius = 0.5, color= color.yellow)


"""
marlough = sphere(pos = (0.62953,-1.1323,0.038369), radius = 0.1, color = color.red)
marlough_vel = vector(0.0154,0.0099574,0.0033815)/k #velocity
marlough_trail = curve(color = color.red)
earth = sphere(pos = (1,0,0), radius = 0.2, color = color.cyan)
earth_vel = vector(0,1,0)
earth_trail = curve(color = color.cyan)
"""

data = np.loadtxt('solar_system.txt',unpack=True)
delta_t = 0.01
obj = []
obj_vel = []
trails = []

for i in data:
    obj.append(sphere(pos = (i[0],i[1],i[2]), radius = 0.1, color = color.blue))
    obj_vel.append(vector(i[3],i[4],i[5])/k)
    trails.append(curve(color = color.blue))

while True:
    rate(10000)
    for o,vel,trail in zip(obj,obj_vel,trails):
        accel = -o.pos/(mag(o.pos)**3)
        vel += accel * delta_t
        o.pos += vel * delta_t
        trail.append(pos=o.pos,retain=50000)
"""
    earth_accel = -earth.pos/(mag(earth.pos)**3)
    earth_vel += earth_accel * delta_t #new velovity = old velocity + acceleration * time
    earth.pos += earth_vel * delta_t #new position = old position + velocity*time
    earth_trail.append(pos=earth.pos,retain = 50000)

    marlough_accel = -marlough.pos/(mag(marlough.pos)**3)
    marlough_vel += marlough_accel * delta_t #new velovity = old velocity + acceleration * time
    marlough.pos += marlough_vel * delta_t #new position = old position + velocity*time
    marlough_trail.append(pos=marlough.pos,retain = 50000)
"""
