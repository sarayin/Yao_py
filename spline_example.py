import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.arange(0,10,1)
y = np.random.uniform(0,10,10)

plt.ion()
plt.figure()
plt.plot(x,y,'o')
sp = interp1d(x,y,1)
xnew = np.linspace(0,9,1000)
ynew = sp(xnew)
plt.plot(xnew,ynew)
