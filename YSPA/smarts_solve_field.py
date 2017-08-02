from astropy.io import fits
import glob
"""
For SMARTS data that starts with 'rccd', Astrometry.net would return an erro
when trying to solve the image due to the missing header values in the FITS file
This batch script removes the missing fits headers without modifying other
information of the file so that Astrometry.net would solve.

Replace the ditrectory, prefix and suffix with the path to your data and run the code.

Come bother Yao if there are any questions.
"""
dir = "/home/student/data/sysygeniuses/20170714SMARTS/solved/"
prefix = "rccd170713."
suffix = ".fits"
files = glob.glob(dir+prefix+"*"+suffix)

for filename in files:
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    header.remove('PIXXMIT')
    header.remove('PIXOFFST')
    header.remove('IRFILTID')
    fits.writeto(filename,data, header,clobber = True)
