import numpy as np
import matplotlib.pyplot as plt
from math import *

def e_taylor_expansion(x, order):
    return sum(x**i/factorial(i) for i in range(order+1))

x = np.linspace(-2, 10, 1e4)
y = np.exp(x)

fig = plt.figure() #Make a matplotlib figure, which can hold plots.

ax = plt.subplot(111)
ax.plot(x, y, label='exp(x)')
ax.plot(x, e_taylor_expansion(x, 1),  label="1st order")
ax.plot(x, e_taylor_expansion(x, 3), label="3rd order")
ax.plot(x, e_taylor_expansion(x, 5),  label="5th order")
ax.plot(x, e_taylor_expansion(x, 7),  label="7th order")
ax.plot(x, e_taylor_expansion(x, 21),  label="21st order")
ax.set_xlim(min(x), max(x))
ax.set_ylim(-2, 200)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Taylor Series Approximations to exp(x) at x = 0")
#plt.semilogy()
plt.legend()
plt.tight_layout()
plt.show(ax)
