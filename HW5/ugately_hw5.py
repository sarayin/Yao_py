# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:06:24 2017

@author: ursulagately
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.nonparametric.kernel_density import KDEMultivariate


data1 = np.loadtxt('Data1.txt')
data2 = np.loadtxt('Data2.txt')

#getting mean and std
mean1 = np.mean(data1)
mean2 = np.mean(data2)
std1 = np.std(data1)
std2 = np.std(data2)

#printing results
print 'Mean of first dataset = %.3f' % mean1
print 'Mean of second dataset = %.3f' % mean2
print 'Standard Deviation of first dataset = %.3f' % std1
print 'Standard Deviation of second dataset = %.3f' % std2

#Cross-Validation of KDE
CV1 = KDEMultivariate(data=data1,var_type='c',bw='cv_ls')
CV2 = KDEMultivariate(data=data2,var_type='c',bw='cv_ls')
print 'Cross-validation gives us this bandwidth for data1 = %.3f' % CV1.bw
print 'Cross-validation gives us this bandwidth for data2 = %.3f' % CV2.bw

#Use silverman's rule of thumb 
Silvrule1 = np.std(data1)*(4./(3.*len(data1)))**(1./5.)
Silvrule2 = np.std(data2)*(4./(3.*len(data2)))**(1./5.)
print 'Silvermans rule of thumb gives us this bandwidth for data1 = %.3f' % Silvrule1
print 'Silvermans rule of thumb gives us this bandwidth for data2 = %.3f' % Silvrule2

#for the first dataset Silverman's rule of thumb and cross validation were somewhat close
#in their given bandwidths (given the value of the data points)
#For the second dataset Silverman's rule of thumb and cross validation were much closer 
#to eachother in the bandwidths they produced (the two varied by roughly 0.02)

#Anderson-Darling test-- determining normality of data
data1_ad = normal_ad(data1)[1]
print 'Anderson-Darling test result (data1 is consistent with being drawn from a normal distribution) = %.5f' % data1_ad

#making histogram for dataset1
plt.ion()
plt.figure(1)
plt.clf()
plt.hist(data1, bins=15,normed=True,edgecolor='none',alpha=0.5, label='Histogram')
plt.xlabel('Data Values')
plt.ylabel('Relative Frequency')
#plotting PDF2
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y
x,y=gauss(sig=std1,x0=mean1)
plt.plot(x,y, linewidth=2, color='r',label='PDF') 
#plotting KDE1
x_grid1 = np.linspace(0,100,1000)
plt.plot(x_grid1,CV1.pdf(x_grid1),color='black', alpha=0.5, lw=2, label='KDE')
#legend and title
plt.legend(loc='upper right')
plt.title('Data 1')

#making histogram for dataset2
plt.figure(2)
plt.clf()
plt.hist(data2, bins=15,normed=True,edgecolor='none',alpha=0.5, label='Histogram')
plt.xlabel('Data Values')
plt.ylabel('Relative Frequency')
#plotting PDF2
mu=1.0
sigma=.2
x = np.linspace(mu-20*sigma,mu+20*sigma,1000)
y = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/(x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, y, linewidth=2, color='r',label='PDF')
#plotting KDE
x_grid2 = np.linspace(0,5,1000)
plt.plot(x_grid2,CV2.pdf(x_grid2), color='black', alpha=0.5, lw=2, label='KDE')
#legend and title
plt.legend(loc='upper right')
plt.title('Data 2')

#Reasoning for why µ=1 and standard deviation is 0.2 for the second dataset's pdf (kstest)
d,p2 = ks_2samp(data2,y)
p2 = 100*p2
print 'The likelihood that they are consistent in being drawn from the same parent distribution = %.5f' % p2 + '%'
#Since the pval is very small, the data is not consistent with being drawn from the same parent distribution


# js comments
#------------------------------
# Nice looking plots!
#
# Great looking code with good commenting.
#
# Text output could be a little better organized.
#
# How could you prevent the RuntimeWarning?
#
# I don't think using ks_2samp is the right function to use if your input is y
#
# Should probably use "probability" instead of "likelihood" now that we have a specific
# definition of the word 'likelihood'
#
# Well done!
#
# 48/50
