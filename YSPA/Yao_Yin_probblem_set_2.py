import numpy as np
import time
def degrees_to_dms(degrees):
    """
    Parameters:
    ----------
    degrees: float
        decimal degrees

    Returns:
    --------
    String: Sexagecimal degrees in the form of 'hh:mm:ss'
    """
    totalSeconds = degrees * 60. * 60.
    seconds = totalSeconds % 60
    minutes = (totalSeconds / 60.) % 60
    degrees = totalSeconds // (60. * 60.)
    return '%.d:%.d:%.2f' % (degrees,minutes,seconds)

def LST(Longitude, JulianDate):
    """
    Parameters:
    ----------
    Longitutde: float
        Longitude of the observer in decimal degrees, East Positive
    Julian Date: float
        Julian Date number

    Returns:
    --------
    String: Local sidereal time at given Julian day in Sexagecimal degrees
    """
    #January 1, 2016 at 00:00 UT = Julian Date 2457388.5 = Local Sidereal Time on the Prime Meridian (longitude = 0) is equal to 06:40:21 (6.6725)
    # 1 solar day = 1.0027379 sidereal days
    # Psuedo code:
    GMTdelta = JulianDate - 2457388.5
    GSTdelta = GMTdelta * 1.0027379
    GSThours = (GSTdelta - int(GSTdelta))*24
    if GSThours + 6.6725 > 24. :
        GST = GSThours + 6.6725 - 24
    else:
        GST = GSThours + 6.6725 #Gives us the sidereal time at Greenwich on the given Julian Day
    LST = GST + Longitude/15
    if 0 <= LST < 24.:
        return degrees_to_dms(LST)
    else:
        return degrees_to_dms(LST+24.)

print 'The local sidereal time at Julian Day 2457570.888889 in New Havens is', LST(-72.9279, 2457570.888889)
