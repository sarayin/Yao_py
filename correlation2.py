"""
Correlation example code with pretty plots
anscombe dataset can be found in https://gitlab.com/jonswift/DataSci
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats, interpolate

"""
This batch script...
1. Wrangles the data sets in anscombe_data.csv
2. Calculate pearson's r for all three data sets
3. Graph scattered plots and best fit lines for all three data sets with annotations

The pearson's r p-value is roughly the probability of two uncorrelated datasets
that have a Pearson correlation at least as extreme as the one computed from these datasets.
reference: Scipy documentation
"""
#Data Wrangling:
data = pd.read_csv('/Users/sara/python/DataSci/anscombe_data.csv',sep=',',
                    skiprows=[0],header=0,names=['xa','ya','xb','yb','xc','yc','xd','yd'])
xa = data['xa']
ya = data['ya']
xb = data['xb']
yb = data['yb']
xc = data['xc']
yc = data['yc']
xd = data['xd']
yd = data['yd']

#Calculating Pearson's r coefficient and p value
ra,pa = stats.pearsonr(xa,ya)
rb,pb = stats.pearsonr(xb,yb)
rc,pc = stats.pearsonr(xc,yc)
rd,pd = stats.pearsonr(xd,yd)

#Plotting:
plt.clf()
plt.ion()
fig = plt.figure(0,figsize=(8,8))

ax1 = fig.add_subplot(221)
fa = np.poly1d(np.polyfit(xa, ya, 1))
xaxisa = np.linspace(min(xa),max(xa),100,endpoint=True)
ax1.plot(xa, ya, 'ro', label='Scatter plot of data a')
ax1.plot(xaxisa, fa(xaxisa), 'm-',label='Best fit line of data a')
ax1.annotate("Pearson's r coefficient = %.5f \n P-value = %.5f" %(ra,pa),[10,280],
             horizontalalignment='left',xycoords='axes pixels',fontsize=8,bbox=dict(boxstyle='round', fc='w'))
ax1.set_ylim(0,15)
ax1.legend(loc='lower right',fontsize=8)

ax2 = fig.add_subplot(222)
fb = np.poly1d(np.polyfit(xb, yb, 1))
fb2 = interpolate.interp1d(xb, yb, kind='quadratic')
xaxisb = np.linspace(min(xb),max(xb),100,endpoint=True)
ax2.plot(xb, yb, 'bo', label='Scatter plot of data b')
ax2.plot(xaxisb, fb(xaxisb), 'c-',label='Best fit linear line of data b')
ax2.plot(xaxisb, fb2(xaxisb), 'k-',label='Best fit quadratic curve of data b')
ax2.annotate("Pearson's r coefficient = %.5f \n P-value = %.5f" %(rb,pb),[10,280],
             horizontalalignment='left',xycoords='axes pixels',fontsize=8,bbox=dict(boxstyle='round', fc='w'))
ax2.set_ylim(0,15)
ax2.legend(loc='lower right',fontsize=8)

ax3 = fig.add_subplot(223)
fc = np.poly1d(np.polyfit(xc, yc, 1))
outlier = xc.index.isin([1])
fc2 = np.poly1d(np.polyfit(xc[~outlier], yc[~outlier], 1))
xaxisc = np.linspace(min(xc),max(xc),100,endpoint=True)
ax3.plot(xc, yc, 'go', label='Scatter plot of data c')
ax3.plot(xaxisc, fc(xaxisc), 'y-',label='Best fit line of data c')
ax3.plot(xaxisc, fc2(xaxisc), 'k-',label='Best fit line of data c excluding the outlier')
ax3.annotate("Pearson's r coefficient = %.5f \n P-value = %.5f" %(rc,pc),[10,280],
             horizontalalignment='left',xycoords='axes pixels',fontsize=8,bbox=dict(boxstyle='round', fc='w'))
ax3.set_ylim(0,15)
ax3.legend(loc='lower right',fontsize=7)

ax4 = fig.add_subplot(224)
fd = np.poly1d(np.polyfit(xd, yd, 1))
outlier = xd.index.isin([1])
fd2 = np.poly1d(np.polyfit(xd[~outlier], yd[~outlier], 1))
xaxisd = np.linspace(min(xd),max(xd),100,endpoint=True)
ax4.plot(xd, yd, 'go', label='Scatter plot of data c')
ax4.plot(xaxisc, fd(xaxisd), 'y-',label='Best fit line of data c')
ax4.plot(xaxisc, fd2(xaxisd), 'k-',label='Best fit line of data c excluding the outlier')
ax4.annotate("Pearson's r coefficient = %.5f \n P-value = %.5f" %(rd,pd),[10,280],
             horizontalalignment='left',xycoords='axes pixels',fontsize=8,bbox=dict(boxstyle='round', fc='w'))
ax4.set_ylim(0,15)
ax4.legend(loc='lower right',fontsize=7)
plt.show()
