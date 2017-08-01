import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from math import *

def degrees_to_dms(angle):
     degrees = int(angle)
     arcmin = int((angle - int(angle))*60.0)
     arcsec = (((angle - int(angle))*60.0) - arcmin)*60.0
     return '%.d:%.d:%.2f' % (degrees,arcmin,arcsec)
#The function that will convert the Ra and Dec from JPL horizons to decimal degrees
def dms_to_degrees(dms):
    degrees,arcmin,arcsec = [float(i) for i in dms.split(':')]
    if degrees < 0.0:
          return degrees - (arcmin/60.0) - (arcsec/3600.0)
    else:
          return degrees + (arcmin/60.0) + (arcsec/3600.0)

def linear_least_squares(x, y, plot =True, sigma_rejection = False):
    """
    Takes 2d arrays and fit the best line to it based on linear Least Squares Reduction formulas.

    Parameters:
    ----------
    x: numpy array
    y: numpy array
    plot: bool
    sigma_rejection: bool or int or float
        specifies the range for sigma rejection, default as False

    Returns:
    --------
    Best fit parameters m,b
    If specified sigma rejection: m, b, and the new parameters after eliminating outliers.

    Example:
    --------
    x = np.array([1.1,1.6,2.0,2.1,2.9,3.2,3.3,4.4,4.9])
    y = np.array([72.61,72.91,73.00,73.11,73.52,73.70,76.10,74.26,74.51])

    m, b, sig, newm, newb, newsig= linear_least_squares(x,y,sigma_rejection=3)
    print 'The most probable y value at x = 3.5 is %.2f plus or minus %.2f ' %(m*3.5+b, sig)
    print 'The new m, b is:',(newm,newb)
    """
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

def lls_plate_solution(file,coords=False,x=False,y=False,ra=False,dec=False,verbose=False):
    """
    Solves for the relationship between x-y coordinates and RA-Dec in an image.

    Parameters:
    ----------
    file: path to text file
        columns should be specifies in the order of ra, dec, x, y
    coords: tuple of int or float
        x-y coordinate of object of interest in the image to convert into RA and Dec
    x, y, ra, dec: array like objects
        need to be specified if a path to text file is not given
    verbose: bool
        prints out results, default as False

    Returns:
    --------
    RA parameters, Dec parameters, RA residuals, Dec residuals,
    RA standard deviation, Dec standard deviation,
    corresponding RA and Dec coordinates if coords is specified.

    Example:
    --------
    ra_params,dec_params,ra_res,dec_res,sigma_ra,sigma_dec, coords_RA, coords_Dec= \
    lls_plate_solution('lspr_marlough.txt',coords = (570.46462,563.53645),verbose=True)
    test_data = lls_plate_solution('lspr_test.txt',coords=(357.62,324.02))

    """
    if file:
        ra, dec, x, y = np.loadtxt(file,unpack=True)
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
    ra_res = ra-(ra_params[0,0]+ra_params.item[0,1]*x+ra_params.item[0,2]*y)
    dec_res= dec-(dec_params[0,0]+dec_params.item[0,1]*x + dec_params[0,2]*y)
    sigma_ra = sqrt((sum(ra_res**2))/(len(x)-3))
    sigma_dec = sqrt((sum(dec_res**2))/(len(x)-3))
    if verbose:
        print 'RA = ',ra_params[0,0],'+',ra_params[0,1],' * x + ',ra_params[0,2],' * y'
        print 'DEC = ', dec_params[0,0],'+',dec_params[0,1],' * x + ',dec_params[0,2],' * y'
        print 'Residuals:'
        print '---------'
        for i in range(len(x)):
            print 'Star #',i+1,':',' RA ->',ra_res[i],' Dec ->',dec_res[i]
        print 'Sigma RA = ',sigma_ra
        print 'Sigma Dec = ',sigma_dec
    if coords:
        x,y = coords
        coords_RA = ra_params[0,0]+ra_params.item[0,1]*x+ra_params.item[0,2]*y
        coords_Dec = dec_params[0,0]+dec_params.item[0,1]*x + dec_params[0,2]*y
        return ra_params,dec_params,ra_res,dec_res,sigma_ra,sigma_dec, coords_RA, coords_Dec
    else:
        return ra_params,dec_params,ra_res,dec_res,sigma_ra,sigma_dec

def centroid(array, center = True):
    """
    Solves for the centroid of an array using the light-weighted mean method.

    Parameters:
    ----------
    array: array like objects
    center: bool
        if True, the center of the image is the (0,0) coordinate
        if False, the top left corner of the image is the (0,0) coordinate
        Default as True

    Returns:
    --------
    x,y coordinates of the centroid

    Example:
    --------
    array = np.array([[0,33,21,33,8],
                         [0,56,51,53,26],
                         [23,120,149,73,18],
                         [55,101,116,50,16],
                         [11,78,26,2,10]])
    centroid(array,center=True)
    """
    M = np.sum(array)
    if center:
        x_low = -(array.shape[1]/2)
        x_high = x_low + array.shape[1]
        y_low = -(array.shape[0]/2)
        y_high = y_low + array.shape[0]
        y_indices, x_indices = np.mgrid[y_low:y_high,x_low:x_high]
        y_indices = y_indices[::-1]
    else:
        x_indices, y_indices = np.indices(array.shape)
    x = np.sum(array*x_indices) / float(M)
    y = np.sum(array*y_indices) / float(M)
    return x, y

def solve_star(file,xcent,ycent,size,plot=True):
    """
    Solves for the centroid and flux of a star in a fits image.

    Parameters:
    -----------
    file: path to fits file
    xcent: float or int
        x coordinate of the estimated center of the star
    ycent: float or int
        y coordinate of the estimated center of the star
    size: int
        size of the square to evaluate surrounding the sta
    plot: bool
        plots the image and the solved star, default as True

    Returns:
    --------
    center of the sub image, centroid of the star in x-y coordinates of the image, flux of the star

    Example:
    --------
    solve_star("2008NU_Spain_2017-07-13_1.fit", 228, 536, 30)
    """
    hdu = fits.open(file)[0]
    image = hdu.data
    star = image[ycent-size:ycent+size,xcent-size:xcent+size]
    sky = image[ycent-size-1:ycent+size+1,xcent-size-1:xcent+size+1]
    skyimg = sky[np.where(sky!=star)]
    sky_val = np.mean(skyimg)
    subimg = star - sky_val
    flux = np.sum(subimg)
    center = centroid(subimg, center = False)
    centroid_x = xcent-size + center[0]
    centroid_y = ycent-size + center[1]
    if plot:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
        ax1.set_title('Full image')
        ax2.set_title('Star centroid')
        ax1.imshow(image, interpolation = 'none', vmin = 700, vmax = 5000, cmap = 'gray')
        ax2.imshow(star, interpolation = 'none', vmin = 700, vmax = 5000, cmap = 'gray')
        ax1.plot(centroid_x,centroid_y,'r+',markersize=10)
        ax2.plot(center[0],center[1],'r+',markersize=10)
        plt.show(fig)
    return center, (centroid_x, centroid_y), flux


def equ_toecl(ra,dec):
    e = np.radians(23.43713) #tilt of the earth in 2017
    #ra, dec = '20 28 07.25', '-14 56 18.2' #Ra, Dec found on JPL horizons
    #Convert Ra and Dec to decimal degrees/radians
    ra = np.fromstring(ra,sep=' ')
    dec = np.fromstring(dec,sep = ' ')
    ra = np.radians(dms_to_degrees(ra[0],ra[1],ra[2])*15.)
    dec = np.radians(dms_to_degrees(dec[0],dec[1],dec[2]))
    #Therefore, the unit vector in the equatorial coordinate system is:
    v_equ = vector(cos(ra)*cos(dec),sin(ra)*cos(dec),sin(dec))

    #Using the multiplication of vectors, we can convert the vector in the equtorial system to the ecliptic system
    trans_matrix = np.array([[1,0,0],[0,cos(e),sin(e)],[0,-sin(e),cos(e)]])
    v_ecl = vector(np.matmul(trans_matrix, v_equ))
    #We can also use the rotate function in vPython to generate a unit vector in the ecliptic coordinate system
    r_prime = rotate(v_equ,-e,axis = vector(1,0,0)) #This vector should be the same as v_ecl

    #Then we can use the vector to find the ecliptic longitude and latitude of our asteroid
    lat = np.arcsin(v_ecl.z)
    lon = np.arcsin(v_ecl.y/cos(lat))
    if lon < 0:
        lon += 360
    ra, dec, lon, lat = degrees(ra), degrees(dec), degrees(lon), degrees(lat)
    if plot:
        x_axis = arrow(pos=(0,0,0),axis = (2,0,0),shaftwidth =0.01, color = color.blue)
        y_axis = arrow(pos=(0,0,0),axis = (0,2,0),shaftwidth =0.01, color = color.green)
        z_axis = arrow(pos=(0,0,0),axis = (0,0,2),shaftwidth =0.01, color = color.red)
        earth = sphere(pos = (0,0,0), radius = 2, color = color.green, opacity = 0.1)
        horizon = cylinder(pos = (0,0,0), axis = (0,0,0.01), radius = 2, opacity = 0.5, color = color.yellow)
        ecliptic = cylinder(pos = (0,0,0), axis = rotate(horizon.axis, e,axis = vector(1,0,0)), radius = 2, opacity = 0.5)
        # 2017-Jul-14 04:00  20 28 14.82 -14 49 14.2
        label(pos = (1,0,0), text = 'x axis', height = 6, box = False, opacity = 0)
        label(pos = (0,1,0), text = 'y axis', height = 6, box = False, opacity = 0)
        label(pos = (0,0,1), text = 'z axis', height = 6, box = False, opacity = 0)
        label(pos = (2,0,0), text = 'vernal equinox', height = 6, box = False, opacity = 0)
        label(pos = (-2,0,2), text = 'Yellow: equatorial, Cyan: ecliptical', box = False, opacity = 0)
        arrow(pos = (0,0,0), axis = v_equ, color = color.yellow)
        arrow(pos = (0,0,0), axis = v_ecl, color = color.cyan)
    return v_equ,v_ecl,lon,lat
