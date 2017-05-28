"""
Get important statistics of a dataset
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:43:34 2017

@author: sara
"""
import numpy as np
import scipy.stats as sc
from statsmodels.nonparametric.kernel_density import KDEMultivariate

data = np.random.normal(1,3,1000)

def weighted_mean(data):
    """
    Return weights and weighted mean for normal data in which each data point has differen variance
    to minimize uncertainty.
    """
    w = []
    for i in data:
        w.append(1./(std(i)**2))
    return w,np.sum(data*w)

def std(data):
    return np.sqrt(np.sum((data-np.mean(data))**2)/(len(data)-1))
def skew(data):
    return (1./std(data)**3)*(1./len(data))*np.sum(((data)-np.mean(data))**3)
def kurtosis(data):
    return (((np.sum(data-np.mean(data)))**4)/(len(data)-1))
def mode(self):
    kde = KDEMultivariate(data, var_type="c", bw="cv_ls")
    pdf = kde.pdf(np.linspace(np.min(data), np.max(data), (np.max(data)-np.min(data))*20))
    return max(pdf), kde.bw

def return_stats(data,weighted_mean=False):
    """
    Return dictionary of important statistics.
    """
    stats = {}
    stats['Mean'] = np.mean(data)
    if weighted_mean:
        stats['Weighted Mean'] = weighted_mean(data)
    stats['Median'] = np.median(data)
    stats['Mode'] = (mode(data)[0])
    stats['RMS'] = np.std(data)
    stats['Standard deviation'] = std(data)
    stats['Variance'] = (std(data)**2)
    stats['Skew'] = skew(data)
    stats['Kurtosis'] = kurtosis(data)
    return stats

# js comments
#------------------------------
# I like the modularity!
#
# Could also define dictionary as such:
# stats = {'Mean': np.mean(data), 'Median':np.median(data), etc }
#
# Your weighted mean is confusing. How are different weights
# calculated for a dataset with no error information?
#
# Procedure gives error when weighted_mean is True
#
# 19/20
"""
George Lawrence
1/24/17
Homework 6
"""
import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate

def getStatistics(arr):
    RMS = np.std(arr)
    med = np.median(arr)
    MAD = np.median(np.abs(arr-np.median(arr)))
    std = standard_deviation(arr)
    skw = skew(arr)
    kurt = kurtosis(arr)
    var = std**2
    mode, bw = modeKDE(arr)
    rn = max(arr)-min(arr)
    return {"RMS": RMS,
    "median": med,
    "MAD": MAD,
    "std": std,
    "skew": skw,
    "kurtosis":kurt,
    "variance": var,
    "mode": mode,
    "bw":bw[0],
    "range":rn
    }

def standard_deviation(arr):
    return np.sqrt(np.sum((arr-np.mean(arr))**2)/(len(arr)-1))

def variance(arr):
    return np.sum((arr-np.mean(arr))**2)/(len(arr)-1)

def kurtosis(arr):
    return (1./standard_deviation(arr)**4)*(1./len(arr))*np.sum((arr-np.mean(arr))**4)-3

def skew(arr):
    return (1./standard_deviation(arr)**3)*(1./len(arr))*np.sum((arr-np.mean(arr))**3)

def modeKDE(arr):
    kde = KDEMultivariate(arr, var_type="c", bw="cv_ml")
    pdf = kde.pdf(np.linspace(min(arr), max(arr), (max(arr)-np.min(arr))*25))
    return max(pdf), kde.bw

# js comments
#------------------------------
# I like the modularity!
#
# Return on modeKDE is a bit sloppy. Also the mode calculation could be sped up
# quite a bit. You don't need the PDF over the whole range.
#
# Nice work!
#
# 20/20
