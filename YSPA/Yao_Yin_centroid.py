import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def centroid(array, center = True):
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


array = np.array([[0,33,21,33,8],
                     [0,56,51,53,26],
                     [23,120,149,73,18],
                     [55,101,116,50,16],
                     [11,78,26,2,10]])

solve_star("/home/student/Yao_py/YSPA/2008NU_Spain_2017-07-13_1.fit", 228, 536, 30)
