import rebound
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from visual import *
import numpy as np
import csv

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
#fig = rebound.OrbitPlot(sim, unitlabel="[AU]", color=True, periastron=True)
#plt.show()
fig = rebound.OrbitPlot(sim,slices=True,color=True,unitlabel="[AU]",lim=10.,limz=1.)
plt.show(fig)
objects = {0:[0.3,color.yellow],1:[0.1,color.blue],2:[0.2,color.orange],3:[0.05,color.white]}

scene = display(title = 'Simulation of the Sun and Earth',width=1200,height = 700,center = (0,0,0))
scene.autoscale = 0
potatoes = []
trails = []
for i in range(len(parts)):
    r = vector(parts[i].x , parts[i].y , parts[i].z)
    rdot = vector(parts[i].vx , parts[i].vy , parts[i].vz)
    potato = sphere(pos = r, radius = objects[i][0] , color = objects[i][1])
    potatoes.append(potato)
    trail = curve(pos = r,color = objects[i][1])
    trails.append(trail)
sim.move_to_com()
#r_snap = np.array([])
earth_potato_dist = []
dt_snap = 10.*k
t_final = 100000. *year
times = []
while sim.t < t_final:
    rate(100000)
    sim.integrate(sim.t + dt_snap)
    times.append(sim.t)
    dist = np.sqrt(np.square(parts[3].x-parts[1].x)+np.square(parts[3].y-parts[1].y)+np.square(parts[3].z-parts[1].z))
    earth_potato_dist.append(dist)
    for particle, potato, trail in zip(sim.particles, potatoes, trails):
        rate(1000)
        r = (particle.x, particle.y, particle.z)
        potato.pos = r
        trail.append(r,retain = 5000)

times = np.array(times)
earth_potato_dist = np.array(earth_potato_dist)
data = np.hstack((times.reshape(len(times),1),earth_potato_dist.reshape(len(earth_potato_dist),1)))
with open("short_term","wb") as f:
    writer = csv.writer(f)
    writer.writerows(data)

#df = pd.read_csv("Long_term",names= ['times','dist'])
#min(df['dist']) = 0.33060002965231272
#print earth_potato_dist
#plt.plot(times,earth_potato_dist)
#plt.show()
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
