"""
George Lawrence
DataSci HW 5
1/23/16
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.stats.diagnostic import normal_ad
from scipy.stats import kstest
import matplotlib.patheffects as pe

# Load data
data1 = np.loadtxt('Data1.txt')
data2 = np.loadtxt('Data2.txt')

# Plot data as normalized histogram
    # Data1
plt.ion()
plt.figure(1)
plt.clf()
bins1 = np.linspace(0,60,20)
plt.hist(data1, alpha=.5, color="r", normed=True, bins=bins1)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Normalized Histogram of Data 1")
plt.show()

    # Data2
plt.figure(2)
plt.clf()
bins2 = np.linspace(1.5,4.5,20)
plt.hist(data2, alpha=.5, color="b", normed=True, bins=bins2)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Normalized Histogram of Data 2")
plt.show()

# KDE with KDEMultivariate
    # Cross validation
data1_KDE_cv = KDEMultivariate(data1, var_type="c", bw="cv_ls")
data2_KDE_cv = KDEMultivariate(data2, var_type="c", bw="cv_ls")

    # Silverman's rule of thumb
data1_KDE_srt = KDEMultivariate(data1, var_type="c", bw="normal_reference")
data2_KDE_srt = KDEMultivariate(data2, var_type="c", bw="normal_reference")
print "--------------------- Kernel Bandwidth ---------------------"
print "Data 1 kernel standard deviation cross validation, Silverman's rule of thumb = %.3f, %.3f" %(data1_KDE_cv.bw[0], data1_KDE_srt.bw[0])
print "Data 2 kernel standard deviation cross validation, Silverman's rule of thumb = %.3f, %.3f" %(data2_KDE_cv.bw[0], data2_KDE_srt.bw[0])

print "--------------------- Data 1 ---------------------"
# Anderson-Darling Test to determine normality of data
data1_ad = normal_ad(data1)[1]
print "Anderson-Darling test p-val for data1: %.5f\n    (Data1 is consistent with being normal distribution)" %(data1_ad)
# Find pdf of data1, then make data using data1 mean and std
x1 = np.linspace(-20,80, 1000)
data1_pdf = data1_KDE_cv.pdf(x1)
    # Parent should be gaussian
data1_parent = 1./(np.std(data1)*np.sqrt(2*np.pi))* np.exp(-1*(x1-np.mean(data1))**2/(2*np.std(data1)**2))

# Plot pdf, data1, and parent
plt.figure(3)
plt.clf()
plt.hist(data1, alpha=.5, color="r", normed=True, bins=bins1, label="Data1 Histogram")
plt.plot(x1, data1_pdf, color="b", path_effects=[pe.Stroke(linewidth=7,foreground="white"), pe.Normal()], label="Data1 KDE", linewidth=5)
plt.plot(x1, data1_parent, color="g", path_effects=[pe.Stroke(linewidth=7,foreground="white"), pe.Normal()],label="Data1 Mean/STD Normal", linewidth=5)
plt.legend(loc="upper left")
plt.xlim(-20,70)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Data1 Histogram, Normal Distribution, and KDE")
plt.show()
print "The KDE compared to the normal distribution made with Data1's mean and standard distribution has a greater mean and standard deviation"

print "--------------------- Data 2 ---------------------"
# Determine if data2 was drawn from lognormal parent
testlognorm = np.random.lognormal(1.,.2)
data2_ks = kstest(data2,"lognorm", args=(np.e,.2))[1]
print "Kolmogorov-Smirnov test p-val for data2: ",data2_ks,"\n  (100 percent certainty the data are not consistent with being drawn from a lognormal parent distribution)"

# Get kernel and parent pdf of data2
x2 = np.linspace(-5, 5,1000)
data2_pdf = data2_KDE_cv.pdf(x2)
data2_parent = 1./(x2*np.sqrt(2*np.pi)*.2) *np.exp(-1*(np.log(x2)-1.)**2/(2*.2**2))

# Plot data2 with parent pdf and KDE
plt.figure(4)
plt.clf()
plt.hist(data2, alpha=.5, color="r", normed=True, bins=bins2, label="Data2 Histogram")
plt.plot(x2, data2_pdf, color="b",path_effects=[pe.Stroke(linewidth=7,foreground="white"), pe.Normal()], label="Data2 KDE", linewidth=5)
plt.plot(x2, data2_parent, color="g", path_effects=[pe.Stroke(linewidth=7,foreground="white"), pe.Normal()], label="Data2 Mean/STD Lognormal", linewidth=5)
plt.legend(loc="upper left")
plt.xlim(-1, 5)
plt.xlabel("Data Values")
plt.ylabel("Relative Frequency")
plt.title("Data2 Histogram, Lognormal Distribution, and KDE")
plt.show()
print "--------------------- The End ---------------------"


# js comments
#------------------------------
# Nice aesthetics on the plotting.
# Thoughtful variable naming
# Informative output, self contained exercise.
#
# What do you think the error on data2_parent computation is all about?
# What would you do to prevent this error?
#
# Not sure why you are getting such a low p-val for Dataset 2.
#
# 50/50
