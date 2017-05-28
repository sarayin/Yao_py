import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf,erfc

# Choose the parameters of the Gaussian distribution to use
sigma = 1.0
mu = 0.0

# Create a cumulative distribution function
x = np.linspace(-5,5,1000)
cdf = 0.5*(1+erf((x-mu)/(np.sqrt(2)*sigma)))

# Plot it
plt.ion()
plt.figure(1)
plt.clf()
plt.plot(x,cdf)

# Now draw samples from a normal distribution with the same
# parameters and plot
data = np.random.normal(mu,sigma,100)
plt.figure(2)
plt.clf()
plt.hist(data)

# Create an empirical CDF and plot on same axis as the exact CDF
xs = np.sort(data)
ys = np.arange(1, len(xs)+1)/float(len(xs))
plt.figure(1)
plt.plot(xs,ys,'g-')
# breaks down for multiple x values (e.g. discrete)

# Use a package to do the same thing
from statsmodels.distributions.empirical_distribution import ECDF
cdf = ECDF(data)
plt.plot(cdf.x, cdf.y,'r--')


# Calculate the K-S statistic and associated probability
from scipy.stats import kstest,ks_2samp
d,p = kstest(data,cdf='norm',args=(mu,sigma))

