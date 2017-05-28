"""
Correlation example code
Shark attack data can be found in HW7 file on https://gitlab.com/jonswift/DataSci
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats
from statsmodels.discrete.discrete_model import Poisson

"""
This code...
1. Reads Shark Attack data
2. Calculates the correlation between the number of shark attacks (total # in the world & in California) and the year
3. Plots the data with appropriate annotations
4. Returns the Pearson's r coefficient and a p-value, which roughly estimates the probability that two uncorrelated datasets
would have a Pearson's r coefficient at least as extreme as the one computed from the two given data sets.
"""

data = pd.read_csv('/Users/sara/python/DataSci/HW7/sharkattacks.csv',skiprows=[0],sep=',',header=0,names=['Year','World','Florida','Australia','Hawaii',
                   'S. Africa','S. Carolina','California','N. Carolina','Reunion','Brazil','Bahamas'])

plt.ion()
fig = plt.figure(0,figsize=(12,4))
plt.clf()

ax1 = fig.add_subplot(131)
keys = data.keys()
for key in keys[1:]:
        ax1.plot(data['Year'],data[key],'o-',label=key,linewidth=1.5)
ax1.set_xlabel("Year", fontsize = 9)
ax1.set_ylabel("Attacks per Year", fontsize = 9)
ax1.set_title("Unprovoked Shark Attacks", fontsize = 9)
ax1.set_xlim([2007, 2016])
ax1.set_ylim([0, 100])
ax1.legend(loc = "right", ncol = 2, fontsize=7)

print '**********************Shark attacks in the world**********************'
ax2 = fig.add_subplot(132)
year = data['Year']
world = data['World']
r2,p_val2 = stats.pearsonr(year,world)
ax2.plot(year,world,'ro')
res2 = Poisson(world,np.ones_like(world)).fit(method='bfgs')
mn2 = np.mean(res2.predict())
ax2.axhline(y=mn2,ls='--',color='g',label='Mean')
ax2.legend(loc='lower right',fontsize=7)
ax2.set_title("Shark attacks in the world", fontsize = 9)
ax2.annotate("Pearson's r coefficient = %.5f \n P-value = %.5f" %(r2,p_val2),[10,280],
             horizontalalignment='left',xycoords='axes pixels',fontsize=7,bbox=dict(boxstyle='round', fc='w'))
print 'The probablity that two unrelated datasets with a mean of %.3f produces an r at least as extream as %.3f is %.3f' %(mn2, r2, p_val2)

print '********************Shark attacks in California***********************'
ax3 = fig.add_subplot(133)
california = data['California']
r3,p_val3 = stats.pearsonr(year,california)
ax3.plot(year,california,'go')
res3 = Poisson(california,np.ones_like(california)).fit(method='bfgs')
mn3 = np.mean(res3.predict())
ax3.axhline(y=mn3,ls='--',color='r',label='Mean')
ax3.legend(loc='lower right',fontsize=7)
ax3.set_title("Shark attacks in California", fontsize = 9)
ax3.annotate("Pearson's r coefficient = %.5f \n P-value = %.5f" %(r3,p_val3),[10,280],
             horizontalalignment='left',xycoords='axes pixels',fontsize=7,bbox=dict(boxstyle='round', fc='w'))
print 'The probablity that two unrelated datasets with a mean of %.3f produces an r at least as extream as %.3f is %.3f'%(mn3, r3, p_val3)


# js comments
#------------------------------
#
# Nice exploration of plotting functions!
#
# Could get the year labels to be a bit more clear.
#
# 20/20
