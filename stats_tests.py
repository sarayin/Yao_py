"""
This is an example code for the student t test and the ks test
Data file: length_data.csv
It can be found in the HW3 file on https://gitlab.com/jonswift/DataSci
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:45:37 2017

@author: sara
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf,erfc
import robust as rb
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kstest,ks_2samp,ttest_ind

data = pd.read_csv('length_data.csv')

def compareData(name1, name2, plot=False):

    #read data
    data1 = data[data['Name']== name1]['Length']
    data2 = data[data['Name']== name2]['Length']

    #exclude outliers
    mean1, mean2 = rb.mean(data1.values), rb.mean(data2.values)
    std1, std2 = rb.std(data1.values), rb.std(data2.values)
    data1 = data1[data1.values>(mean1-std1)]
    data2 = data2[data2.values>(mean2-std2)]
    #std1 = np.std(lengths1)

    if plot:
        #plot histograms
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.hist(data1.values,bins=np.linspace(4.75,5.75,30),
                     label=name1, alpha=0.5,histtype='barstacked',stacked=True)

        plt.hist(data2.values,bins=np.linspace(4.75,5.75,30),
                     label=name2, alpha=0.5,histtype='barstacked',stacked=True)
        plt.legend()

        #plot CDFs
        plt.figure(2)
        xs1 = np.sort(data1)
        ys1 = np.arange(1, len(xs1)+1)/float(len(xs1))
        plt.plot(xs1,ys1,'g-',label=name1)
        cdf = ECDF(data2)
        plt.plot(cdf.x, cdf.y,'r-',label=name2)
        plt.legend()

    #preform ks test and student t-test
    d,pk = ks_2samp(data1,data2)
    t,pt = ttest_ind(data1,data2,equal_var=False)

    #print d,pk,pt
    return pk, pt

names = np.array(data['Name'].unique()).tolist()

#Calculate which pairs of data are consistent in being drawn from the same parent distribution and/or the same mean using the function above
#and print the results
print '-------------Data Analysis Report-------------'
print '                 **KS test**                  '
print 'The probablity that the measurements of these pairs are consistent with being drawn from the same parent distribution is less than or equal to 5 percent:'
for name1 in names:
    i = names.index(name1)
    for name2 in names[i:]:
       if name2!=name1 and compareData(name1,name2)[0] <= 0.05:
           print name1 + ' and ' + name2 + ' ' + ', p-value of ks test = %.3f' % (compareData(name1,name2)[0])
print 'Note:'
print 'If the p value of the ks test of two data sets is less than 5%, it is low enough to disprove the null hypothesis.'
print 'Therefore, one can say with a greater than 95% confidence that the two data sets are being drawn from different parent distributions.'
print '----------------------------------------------'
print '              **Student t-test**              '
print 'The probablity that the measurements of these pairs are consistent with being drawn from parent distributions with the same mean is less than or equal to 5 percent:'
for name1 in names:
    i = names.index(name1)
    for name2 in names[i:]:
       if name2!=name1 and compareData(name1,name2)[1] <= 0.05:
           print name1 + ' and ' + name2 + ' ' + ', p-value of t-test = %.3f' % (compareData(name1,name2)[1])
print 'Note:'
print 'If the p value of the student t test of two data sets is less than 5%, it is low enough to disprove the null hypothesis.'
print 'Therefore, one can say with a greater than 95% confidence that the two data sets are being drawn from parent distributions with different mean values.'
print '----------------------------------------------'


# js comments
#------------------------------
#
# This is really slick... nice work, Yao!
#
# George seems to be the culprit as he is involved in almost all of the discrepant
# pairs.
#
# On the whole, how consistent were we?
#
# 50/50
