"""
Julien Luebbers
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.stats.diagnostic import normal_ad
from scipy.stats import kstest

data1 = np.loadtxt('Data1.txt')
data2 = np.loadtxt('Data2.txt')

plt.ion()
plt.figure(1)
plt.clf()
plt.hist(data1, bins=20, color="b", normed=True)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Normed Histogram of Data 1")
plt.show()

plt.figure(2)
plt.clf()
plt.hist(data2, bins=20, color="g", normed=True)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Normaled Histogram of Data 2")
plt.show()

#cross valid
data1_KDE_cv = KDEMultivariate(data1, var_type="c", bw="cv_ml")
data2_KDE_cv = KDEMultivariate(data2, var_type="c", bw="cv_ml")
#rule of thumb
data1_KDE_rt = KDEMultivariate(data1, var_type="c", bw="normal_reference")
data2_KDE_rt = KDEMultivariate(data2, var_type="c", bw="normal_reference")
print "Data 1 kernel standard deviation cross validation and rule of thumb respectively: %.3f, %.3f" %(data1_KDE_cv.bw[0], data1_KDE_rt.bw[0])
print "Data 2 kernel standard deviation cross validation and rule of thumb respectively: %.3f, %.3f" %(data2_KDE_cv.bw[0], data2_KDE_rt.bw[0])

"""
This indicates that the cross validation demands a wider kernel than the rule of thumb in both cases although much more so in the first.
"""

d_ad,p_ad = normal_ad(data1)
print "ad test p-val for data1: %.5f\n    Data1 is consistent with being drawn from a normal distribution." %(p_ad)

x1 = np.linspace(0,70,1000)
data1_parent = data1_KDE_cv.pdf(x1)
data1_derived = 1./(np.sqrt(2*np.pi)*np.std(data1))*np.exp(-1*(x1-np.mean(data1))**2/(2*np.std(data1)**2))
parentstd = np.std(data1_parent)
parentmn = np.mean(data1_parent)
print "The mean and standard deviation of data1's parent distribution: %.3f, %.3f" %(parentmn,parentstd)
#create plot for all the stuff to go n

plt.figure(3)
plt.clf()
plt.hist(data1, color="g", normed=True, bins=20, label="Data1 Histogram")
plt.plot(x1, data1_parent, "k-", label="Data1 Parent Distribution PDF", linewidth=3)
plt.plot(x1, data1_derived, "b--",label="Data1 Derived PDF from Sample", linewidth=3)
plt.legend(loc="upper left")
plt.xlim(-20,70)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Data1 Plots")
plt.show()
print "The KDE has a visually smaller standard deviation than the PDF."
# lognormal data2?
data2_ks = kstest(data2,"lognorm", args=(1.,.2))[1]
print "KSTest p_val: %.5f\n  with 100 percent certainty the data are not consistent with being drawn from a lognormal parent distribution" %data2_ks
# same as earlier, only data2.
x2 = np.linspace(-1,5,1200)
data2_parent = data2_KDE_cv.pdf(x2)
data2_derived = 1./(np.sqrt(2*np.pi)*np.std(data2))*np.exp(-1*(x2-np.mean(data2))**2/(2*np.std(data2)**2))

plt.figure(4)
plt.clf()
plt.hist(data2, color="b", normed=True, alpha=.5, bins=20, label="Data2 Histogram")
plt.plot(x2,data2_parent, "k-",label="Data2 Parent Distribution PDF", linewidth=3)
plt.plot(x2,data2_derived, "g--", label="Dist derived from sample", linewidth=3)
plt.legend(loc="upper left")
plt.xlim(-1, 5)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Data2 Plots")
plt.show()

"""
Hist, KDE, and our best guess @ the pdf.
"""

# js comments
#------------------------------
# Beautiful plots!
# Neat code.
# Think about using line continuation character "\" to wrap long lines
#
# I think the input parameters for the lognormal distribution
# are wrong. Check the numpy page for np.random.lognormal.
#
# Variable naming convention a little off. Should call data1_parent, data1_KDE
# as it is the kernel density estimation of the parent.
#
# 49/50
