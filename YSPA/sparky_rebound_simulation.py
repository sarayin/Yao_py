import rebound
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import csv
from math import *

k = 2*pi/365. #modified day per day
year = 2*pi #modified day per year
sim = rebound.Simulation()
date = "2017-08-01 00:00"
sim.add("Sun", date = date)
sim.add("399", date = date)
sim.add("Jupiter", date = date)
#sim.add("2008 NU", date = date)
sim.add(m = 0,x= 0.631695419869991, y=-1.03557552813299, z=-0.406141242151842, \
        vx = 0.885336584655336, vy = 0.462313854222598, vz = 0.398919500273978, date=date)
parts = sim.particles
sim.move_to_com()
earth_potato_dist = []
dt_snap = 1.*k
t_final = 100000. *year
times = []
while sim.t < t_final:
    sim.integrate(sim.t + dt_snap)
    times.append(sim.t)
    dist = np.sqrt(np.square(parts[3].x-parts[1].x)+np.square(parts[3].y-parts[1].y)+np.square(parts[3].z-parts[1].z))
    earth_potato_dist.append(dist)
    print sim.t,dist
times = np.array(times)
earth_potato_dist = np.array(earth_potato_dist)
data = np.hstack((times.reshape(len(times),1),earth_potato_dist.reshape(len(earth_potato_dist),1)))
with open("Long_term","wb") as f:
    writer = csv.writer(f)
    writer.writerows(data)
