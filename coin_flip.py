"""
Posterior function example code
"""
import numpy as np
import matplotlib.pyplot as plt
import random

trails = 6
results = []
for i in np.random.random(trails):
    if i > 0.5: #heads
        results.append(1)
    if i < 0.5: #tails
        results.append(0)

heads = results.count(1)
tails = results.count(0)

#plot:
x = np.linspace(0,1,100)
y = x**heads*(1-x)**tails
plt.ion()
plt.clf()
plt.plot(x,y,'r-',label='Posterior')
plt.annotate('#Trails = %i, #Heads = %i, #Tails = %i' %(trails, heads, tails),xy=[10,10],xycoords='figure pixels')
plt.legend()
plt.show()

# js comments
#------------------------------
# I think you mean trials, not trails :)
#
# Can you rule out your coin being fair given your data? What would
# you expect your curve to do as you increase the number of trials?
#
# 20/20
