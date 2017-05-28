"""
Created on Fri Jan 20 14:17:01 2017

@author: cvyhnal
"""

#   imports math routines, plotting routines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

#   reads both datasets
data1 = np.loadtxt('Data1.txt')
data2 = np.loadtxt('Data2.txt')

#   calcualtes descriptive statistics for data set 1 (hw5.test1-b)
import statsmodels.api as sm 
mean1 = np.mean(data1)              
med1 = np.median(data1)                  
std1 = np.std(data1)

#   histogram of first dataset with assumed Gaussian overlay (hw5.1)
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y
plt.ion()
plt.figure(1)
plt.clf()
plt.title('Histogram of Data Set 1')
plt.hist(data1, normed = True) 
x,y = gauss(x0=np.mean(data1),sig=np.std(data1))
plt.plot(x,y,'r--')
sig = 1.0
xlim = plt.xlim(med1-5*sig,med1+5*sig)
np.sqrt(np.e)

#   produces the StatsModel KDE for data set 1 (hw5.2)
dens1_u = sm.nonparametric.KDEMultivariate(data1, var_type = 'c', bw = 'normal_reference')
dens1_u.bw
print
print 'Results for analysis of data set 1:'
print '-----------------------------------'
print 'Stats Model KDE = %.3f' % dens1_u.bw

#   produces the Silverman's 'rule of thumb' KDE (hw5.3)
rot1 = 1.06*std1*(len(data1))**(-1/5)
print 'Silverman ROT KDE= %.3f' % rot1

#   calcualtes descriptive statistics for data set 1 (hw5.test1-b)
import statsmodels.api as sm 
mean1 = np.mean(data1)              
med1 = np.median(data1)                  
std1 = np.std(data1)

#   is dataset 1 Gaussian? (hw5.test1-a)
k2, pval = sp.stats.mstats.normaltest(data1)
print 'p value for Gaussian = %.3f' % pval
print 'mean = %.3f' % mean1
print '1 sigma standard deviation = %.3f' % std1
print

#   How do I get fit values/results to display on the plot itself, and not in the text window?

#   produces the StatsModel KDE for data set 2 (hw5.2)
dens2_u = sm.nonparametric.KDEMultivariate(data2, var_type = 'c', bw = 'normal_reference')
dens2_u.bw
print
print 'Results for analysis of data set 2:'
print '-----------------------------------'
print 'Stats Model KDE for data set 2 = %.3f' % dens2_u.bw

#   produces the Silverman's 'rule of thumb' KDE (hw5.3)
rot2 = 1.06*std2*(len(data2))**(-1/5)
print 'Silvermans ROT KDE = %.3f' % rot2

med2 = np.median(data2)                  
std2 = np.std(data2)

#   histogram of second dataset (hw5.1)
def lognorm(mu=1, std=0.2):
    x = np.linspace(0,5,1000)
    y = 1.0/(std*x*np.sqrt(2*np.pi))*np.exp((np.log(x)-mu)**2/(2*std**2))
    return x,y
plt.ion()
plt.figure(2)
plt.clf()
plt.title('Histogram of Data Set 2')
plt.hist(data2, normed = True) 
x,y = lognorm(mu=np.mean(data2),std=np.std(data2))
plt.plot(x,y,'r--')
xlim = plt.xlim(0,5)
np.sqrt(np.e)

#   stuck here trying to reproduce theoretical overlay on lognormal dist


# js comments
#------------------------------
# Neat code, and nice commenting.
# 
# There is an error on line 35: sig not defined.
#
# You can use plt.annotate function to put the stat results on the
# plot itself.
#
# Plots need some work. Pilfer some of the other students code from
# the HW5 directory. See Yao's in particular.
#
# You have the right approach for overlaying theoretical distribution
# by creating functions.
