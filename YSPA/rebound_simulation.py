import rebound
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from visual import *
import numpy as np

k = 2*pi/365. #modified day per day
year = 2*pi #modified day per year
sim = rebound.Simulation()
date = "2017-08-01 00:00"
sim.add("Sun", date = date)
sim.add("399", date = date)
sim.add("Jupiter", date = date)
#sim.add("2008 NU", date = date)
sim.add(m = 0, x = 0.704752, y = -1.5479, z = -0.429144, vx=0.51909868, vy=0.63919601, vz=0.15504233, date=date)
parts = sim.particles
#fig = rebound.OrbitPlot(sim, unitlabel="[AU]", color=True, periastron=True)
#plt.show()

scene = display(title = 'Simulation of the Sun and Earth',width=1200,height = 700,center = (0,0,0))
scene.autoscale = 0

potatoes = []
trails = []
for i in range(len(parts)):
    r = vector(parts[i].x , parts[i].y , parts[i].z)
    rdot = vector(parts[i].vx , parts[i].vy , parts[i].vz)
    potato = sphere(pos = r, radius = parts[i].m/2. , color = color.white)
    potatoes.append(potato)
    trail = curve(pos = r,color = color.white)
    trails.append(trail)
sim.move_to_com()
#x_snap,y_snap,z_snap = np.zeros((4,)),np.zeros((4,)),np.zeros((4,))
earth_potato_dist = []
dt_snap = 1.*k
t_final = 10. *year
while sim.t < t_final:
    rate(1000)
    sim.integrate(sim.t + dt_snap)
    xarray, yarray, zarray = np.array([]),np.array([]),np.array([])
    dist = np.sqrt(np.square(parts[3].x-parts[1].x)+np.square(parts[3].y-parts[1].y)+np.square(parts[3].z-parts[1].z))
    earth_potato_dist.append(dist)
    for particle, potato, trail in zip(sim.particles, potatoes, trails):
        rate(1000)
        r = (particle.x, particle.y, particle.z)
        potato.pos = r
        trail.append(r,retain = 5000)
        #xarray = np.append(xarray, particle.x).T
        #yarray = np.append(yarray, particle.y).T
        #zarray = np.append(zarray, particle.z).T
    #x_snap = np.hstack((x_snap, xarray))
    #y_snap = np.hstack((x_snap, yarray))
    #z_snap = np.hstack((x_snap, zarray))
    #print sim.t
print earth_potato_dist
plt.plot(earth_potato_dist)
plt.show()
"""
ssh -X student@sparky.local
cd /path/to/code
sftp student@sparky.local #To put your code on Dr. Faison's computer:
While in sftp, prepend l to any command in terminal to execute it on your computer, e.g:
lls #lists files on your computer
ls #lists files on the remote computer
put filename #copies files from your computer to remote.
get filename #copies files from remote to your computer.
"""
