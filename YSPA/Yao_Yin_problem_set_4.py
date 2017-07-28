import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *
#=====================
# Problem 1
#=====================

def linear_least_squares(x, y, plot =True, sigma_rejection = False):
    try:
        type(x) == np.ndarray and type(y) == np.ndarray
    except:
        print 'Type Error: x and y should be numpy arrays, not lists.'
    m = (x*y-y.mean()*x).sum()/(x**2-x.mean()*x).sum()
    b = y.mean()-m*x.mean()
    res = (m*x)+b-y
    if plot:
        fig,ax = plt.subplots()
        ax.errorbar(x,y,yerr = np.std(res),color='b',fmt='o')
        ax.plot(np.linspace(0,5,10),m*np.linspace(0,5,10) + b,'r--',label = 'Best Fit Line')
        ax.set_title('Linear Least Squares Best Fit Lines')
    if sigma_rejection:
        try:
            type(sigma_rejection)!= str or type(sigma_rejection)!= bool
        except:
            print 'Value Error: sigma_rejection value should be string or float.'
            return
        for i in range(len(x)-1):
            if y[i] > (m*x[i]+b+sigma_rejection*np.std(res)) or \
             y[i] < (m*x[i]+b-sigma_rejection*np.std(res)):
                print 'At x = %.2f, the y value is %.2f, \
                more than %f sigmas away from the prediction. Being rejected.'% (x[i], y[i], sigma_rejection)
                x = np.delete(x,i)
                y = np.delete(y,i)
        newm = (x*y-y.mean()*x).sum()/(x**2-x.mean()*x).sum()
        newb = y.mean()-m*x.mean()
        newres = (newm*x)+newb-y
        ax.plot(np.linspace(0,5,10),newm*np.linspace(0,5,10) + newb,'g--',label = 'Sigma Rejection')
    plt.legend(loc='upper left')
    plt.show(fig)
    if sigma_rejection:
        return m,b,np.std(res),newm,newb,np.std(newres)
    else:
        return m, b

x = np.array([1.1,1.6,2.0,2.1,2.9,3.2,3.3,4.4,4.9])
y = np.array([72.61,72.91,73.00,73.11,73.52,73.70,76.10,74.26,74.51])

m, b, sig, newm, newb, newsig= linear_least_squares(x,y,sigma_rejection=3)
print 'The most probable y value at x = 3.5 is %.2f plus or minus %.2f ' %(m*3.5+b, sig)
print 'The new m, b is:',(newm,newb)
'''
lspr_test = np.loadtxt('lspr_test.txt',unpack = True)
ra, dec, x, y = lspr_test
ra_params,dec_params,ra_res,dec_res,sigma_ra,sigma_dec = lls_plate_solution(x,y,ra,dec,verbose=True)
RA = ra_params.item(0)+ra_params.item(1)*357.62+ra_params.item(2)*324.02
Dec = dec_params.item(0)+dec_params.item(1)*357.62+dec_params.item(2)*324.02
'''
#========================
# Problem 2
# Marlough
# SMARTS rccd data
# Image center coords
# RA = 20 28 18.57,Dec = -14 43 11.6
# JD = 2457948.78960
#========================
def lls_plate_solution(x,y,ra,dec,verbose=False):
    try:
        len(x) == len(y) == len(ra) == len(dec)
    except:
        print "Value Error: Coordinate inputs have diffrent dimensions."
        return
    ra_matrix = np.matrix([ra.sum(),(ra*x).sum(),(ra*y).sum()])
    dec_matrix = np.matrix([dec.sum(),(dec*x).sum(),(dec*y).sum()])
    trans_matrix = np.matrix([[len(x),x.sum(),y.sum()],
                    [x.sum(),(x**2).sum(),(x*y).sum()],
                    [y.sum(),(x*y).sum(),(y**2).sum()]])
    ra_params= ra_matrix*trans_matrix.I
    dec_params = dec_matrix*trans_matrix.I
    ra_res = ra-(ra_params.item(0)+ra_params.item(1)*x+ra_params.item(2)*y)
    dec_res= dec-(dec_params.item(0)+dec_params.item(1)*x + dec_params.item(2)*y)
    sigma_ra = sqrt((sum(ra_res**2))/(len(x)-3))
    sigma_dec = sqrt((sum(dec_res**2))/(len(x)-3))
    if verbose:
        print 'RA = ',ra_params.item(0),'+',ra_params.item(1),' * x + ',ra_params.item(2),' * y'
        print 'DEC = ', dec_params.item(0),'+',dec_params.item(1),' * x + ',dec_params.item(2),' * y'
        print 'Residuals:'
        print '---------'
        for i in range(len(x)):
            print 'Star #',i+1,':',' RA ->',ra_res[i],' Dec ->',dec_res[i]
        print 'Sigma RA = ',sigma_ra
        print 'Sigma Dec = ',sigma_dec
    return ra_params,dec_params,ra_res,dec_res,sigma_ra,sigma_dec

marlough_txt = np.loadtxt('lspr_marlough.txt',unpack=True)
star_ra, star_dec, star_x, star_y = marlough_txt
ra_params,dec_params,ra_res,dec_res,sigma_ra,sigma_dec = lls_plate_solution(star_x,star_y,star_ra,star_dec,verbose=True)

marlough_x = 570.46462
marlough_y = 563.53645

marlough_RA = ra_params.item(0)+ra_params.item(1)*marlough_x+ra_params.item(2)*marlough_y
marlough_Dec = dec_params.item(0)+dec_params.item(1)*marlough_x + dec_params.item(2)*marlough_y
