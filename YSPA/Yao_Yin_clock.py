#=============
# Yao's Clock!
#=============
from visual import *
clock_face = cylinder(pos = vector(0,0,0), radius = 7, axis=vector(0,0,0.01))
hour_hand = arrow(pos=vector(0,0,0), axis = vector(0,1,0), length = 3, shaftwidth=0.2, color = color.orange)
minute_hand = arrow(pos=vector(0,0,0), axis = vector(0,1,0), length = 4, shaftwidth=0.1, color = color.green)
second_hand = arrow(pos=vector(0,0,0), axis = vector(0,1,0), length = 5, shaftwidth=0.05, color = color.red)

d = vector(0,6,0)
for i in range(12):
    d = rotate(d, angle = radians(-30))
    label(pos=d,text = str(i+1), font = 'serif', height = 20, color = color.black, opacity = 0, box = False)
t = 0.0

while t < 600000:
    rate(100)
    second_hand.axis = rotate(second_hand.axis,angle = radians(-6), axis = (0,0,1))
    minute_hand.axis = rotate(minute_hand.axis, angle = radians(-0.1))
    hour_hand.axis = rotate(hour_hand.axis, angle = radians(-0.1/60.))
    t +=1
