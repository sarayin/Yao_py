"""
Example code for kernel density estimation
Original data files can be found in the HW5 file on https://gitlab.com/jonswift/DataSci
To Do:
1. Change the way I use the ks test
2. Maybe modulize
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:05:52 2017
@author: sara
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.stats.diagnostic import normal_ad
from scipy.stats import kstest,ks_2samp,ttest_ind, gaussian_kde
def gauss(sig,x0):
    x = x_grid1
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

def lognormal(sig,x0):
    x = x_grid2
    y =  (np.exp(-(np.log(x)-x0)**2/(2*sig**2))/(x*sig*np.sqrt(2*np.pi)))
    return x,y

data1 = np.loadtxt('Data1.txt')
data2 = np.loadtxt('Data2.txt')

#Calculate bandwidth with Cross Validation Least Seuqares
dens1 = KDEMultivariate(data=[data1], var_type='c', bw='cv_ls')
dens2 = KDEMultivariate(data=[data2], var_type='c', bw='cv_ls')
#Calculate bandwidth with Silverman's rule of thumb
bw1 = np.std(data1)*(4./(3.*len(data1)))**(1./5.)
bw2 = np.std(data2)*(4./(3.*len(data2)))**(1./5.)

#Analyzing Data 1: KDE, Parent distribution, std and mean
x_grid1 = np.linspace(0,70,1000)
pdf1 = dens1.pdf(x_grid1)
mean1, std1 = np.mean(data1),np.std(data1)
x1, y1 = gauss(std1,mean1)
p1 = normal_ad(data1)[1]
mean_kde1, std_kde1 = np.mean(pdf1),np.std(pdf1)
#Analyzing Data 2: KDE, Parent distribution, std and mean
x_grid2 = np.linspace(0,70,1000)
pdf2 = dens2.pdf(x_grid2)
mean2, std2 = np.mean(data2),np.std(data2)
x2,y2 = lognormal(0.2,1.0)
p2 = ks_2samp(y2,data2)[1]
mean_kde2, std_kde2 = np.mean(pdf2),np.std(pdf2)


#Plot the histograms, the parent distributions and the KDEs
plt.ion()
plt.clf()
fig = plt.figure(00,figsize=(15,5))
ax1 = fig.add_subplot(121)
ax1.hist(data1, bins=15, normed=True,edgecolor='none',color='c',alpha=0.7,label='Histogram')
ax1.plot(x_grid1,y1,'r--', lw=2, alpha=0.7,label='Normal Parent Distribution')
ax1.plot(x_grid1,pdf1,color='black', alpha=0.7, lw=2, label='KDE')
ax1.set_title('Data 1')
ax1.set_xlim(-10,80)
ax1.set_ylim(0,0.05)
ax1.legend(loc='upper left',fontsize='small')
ax1.set_xlabel('Data Values')
ax1.set_ylabel('Relative Frequency')

ax2 = fig.add_subplot(122)
ax2.hist(data2, bins=15,normed=True,edgecolor='none',color='m',alpha=0.5, label='Histogram')
ax2.plot(x_grid2,y2, 'r--', label='Lognormal Parent Distribution')
ax2.plot(x_grid2,pdf2, color='black', alpha=0.5, lw=2, label='KDE')
ax2.set_title('Data 2')
ax2.set_xlim(-1,5)
ax2.set_ylim(0,1.6)
ax2.legend(loc='upper left',fontsize='small')
ax2.set_xlabel('Data Values')
ax2.set_ylabel('Relative Frequency')

print '-------------Data Analysis Report-------------'
print '                 By: Yao                      '
print '                 **Data 1**                   '
print 'The kernel width derived using cross-validation is: %.4f' % dens1.bw
print "The kernel width estimated from Silverman's rule of thumb is: %.4f" % bw1
print 'According to the Aderson Darling test, the chance that Data 1 is consistent with being normally distributed is %.4f' % p1
print 'The best estimate for the mean of the parent distribution of Data 1 is %.4f, and the standard deviation is %.4f' % (mean1,std1)
print 'Visually on the graph, the KDE has a slightly larger mean and standard deviation than the normal distribution derived from the mean and std of the sample.'
print '----------------------------------------------'
print '                 **Data 2**                   '
print 'The kernel width derived using cross-validation is: %.4f' % dens2.bw
print "The kernel width estimated from Silverman's rule of thumb is: %.4f" % bw2
print 'According to the KS test, the probablity that Data 2 is consistent with a std=0.2, mu=1.0 lognormal distribution is %s' % str(p2)
print 'Therefore we can say with a almost 100% confidence that Data 2 is not being drawn from the log normal distribution.'
print 'The best estimate for the mean of the parent distribution of Data 2 is %.3f, and the standard deviation is %.4f' % (mean2,std2)
print 'Visually on the graph, the KDE has a slightly larger mean than the lognormal distribution derived from the mean and std of the sample.'
print '----------------------------------------------'

# js comments
#------------------------------
# Cool looking plots!
# I love how you consolidated the money plot into a single figure pane :)
#
# I don't think you've used ks_2samp correctly. The other input needs
# to be a distribution, not the y values of a pdf.
#
# Great comments on the analysis.
#
# 49/50
