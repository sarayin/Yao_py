from visual import *
import numpy as np
from math import *

k = 2*pi/365. #modified day per day

xaixis = arrow(pos=(0,0,0),axis=(1,0,0),shaftwidth=0.03, length = 10, color = color.blue)
yaixis = arrow(pos=(0,0,0),axis=(0,1,0),shaftwidth=0.03, length = 10, color = color.green)
zaixis = arrow(pos=(0,0,0),axis=(0,0,1),shaftwidth=0.03, length = 10,color = color.orange)

sun = sphere(pos = (0,0,0), radius = 0.5, color= color.yellow)

delta_t = 0.01

rk4_obj = sphere(pos = (0.9,0,0), radius = 0.1, color = color.red)
rk4_vel = vector(0,1.3,0) #velocity
rk4_trail = curve(color = color.red)

euler_obj = sphere(pos = (0.9,0,0), radius = 0.1, color = color.blue)
euler_vel = vector(0,1.3,0)
euler_trail = curve(color = color.blue)

def a(vector): #Acceleration due to gravity
    return -vector/(mag(vector)**3)
t = 0

while True:
    rate(1000)
    r1 = rk4_obj.pos
    v1 = rk4_vel
    k1 = a(r1)

    v2 = v1 + k1*delta_t*0.5
    r2 = r1 + v1*delta_t*0.5
    k2 = a(r2)

    v3 = v1 + k2*delta_t*0.5
    r3 = r1 + v2*delta_t*0.5
    k3 = a(r3)

    v4 = v1 + k3*delta_t
    r4 = r1 + v3*delta_t
    k4 = a(r4)

    v_next = v1 + (k1+2*k2+2*k3+k4)/6.0 * delta_t #modified acceleration
    r_next = r1 + (v1+2*v2+2*v3+v4)/6.0 * delta_t

    rk4_obj.pos = r_next
    rk4_vel = v_next
    rk4_trail.append(pos=rk4_obj.pos,retain=50000)

    euler_vel += a(euler_obj.pos) * delta_t
    euler_obj.pos += euler_vel * delta_t
    euler_trail.append(pos=euler_obj.pos,retain=50000)
    t += delta_t
    #if euler_obj.pos.y <= 0.:
    #    print "The object is crossing the x -axis !",euler_obj.pos,euler_vel
    #    print t
    #    break
    if (mag(rk4_obj.pos)-mag(euler_obj.pos))/mag(rk4_obj.pos) > 0.01:
        print "The prediction of the two methods differs by more than 1%!"
        print rk4_obj.pos, euler_obj.pos, t
        break
