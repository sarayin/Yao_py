import numpy as np
from ephemPy import Ephemeris as Ephemeris_BC
from visual import *
import matplotlib.pyplot as plt
from math import *

class Ephemeris(Ephemeris_BC):

    def __init__(self, *args, **kwargs):
        Ephemeris_BC.__init__(self, *args, **kwargs)
        self.AUFAC = 1.0/self.constants.AU
        self.EMFAC = 1.0/(1.0+self.constants.EMRAT)

    def position(self, t, target, center):
        pos = self._position(t, target)
        if center != self.SS_BARY:
            pos = pos - self._position(t, center)
        return pos

    def _position(self, t, target):
        if target == self.SS_BARY:
            return numpy.zeros((3,), numpy.float64)
        if target == self.EM_BARY:
            return Ephemeris_BC.position(self, t, self.EARTH)*self.AUFAC
        pos = Ephemeris_BC.position(self, t, target)*self.AUFAC
        if target == self.EARTH:
            mpos = Ephemeris_BC.position(self, t, self.MOON)*self.AUFAC
            pos = pos - mpos*self.EMFAC
        elif target == self.MOON:
            epos = Ephemeris_BC.position(self, t, self.EARTH)*self.AUFAC
            pos = pos + epos - pos*self.EMFAC
        return pos
ephem = Ephemeris('405')

def toDecimal(degrees, minutes, seconds):
    minutes = np.true_divide(minutes,60)
    seconds = np.true_divide(seconds,3600)
    if degrees < 0:
        return -(-degrees+minutes+seconds)
    return degrees+minutes+seconds

def toSexagesimal(number):
    degrees = np.trunc(number)
    number = 60.0*(number - degrees)
    minutes = np.trunc(number)
    number = 60.0*(number - minutes)
    seconds = number
    if degrees < 0:
        return degrees, -minutes, -seconds
    return degrees, minutes, seconds

def plot(RAs, Decs):
    plt.scatter(RAs, Decs)
    plt.title('Right Ascension v. Declination')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.show()

def toRectangular(alpha, delta):
    unit = vector(np.cos(radians(delta))*np.cos(radians(alpha)),np.cos(radians(delta))*np.sin(radians(alpha)),np.sin(radians(delta)))
    return unit

def f(tau, r, rdot=None):
    if rdot is None:
        f = 1. - (tau**2/(2*((mag(r))**3)))
    else:
        f = 1. - (tau**2/(2*((mag(r))**3))) + (np.dot(r, rdot))*(tau**3)/(2*((mag(r))**5))
    return f

def g(tau, r, rdot=None):
    if rdot is None:
        g = tau - (tau**3)/(6*(mag(r)**3))
    else:
        g = tau - (tau**3)/(6*((mag(r))**3)) + (np.dot(r, rdot))*(tau**4)/(4*((mag(r))**5))
    return g

def toLST(longitude, julian):
    sideadjust = (((julian - 2457388.5)*1.0027379) % 1)*24.0 #sidereal time adjustment
    decimal = 6.6725 #original sidereal time
    pmeridtime = decimal + sideadjust
    if pmeridtime > 24:
        pmeridtime -= 24
    longadjust = longitude/15.0 #longitude conversion
    lst = pmeridtime + longadjust #adjusting for longitude
    if lst < 0:
        lst += 24
    if lst > 24:
        lst -= 24
    return lst*15

def gen_ephem(lat, lon, JD, r):
    #convert JD/lon to lst
    lst = toLST(lon, JD)
    #convert to rectangular coordinates and scale geocentric-NH vector
    earthtoNH = vector(cos(radians(lat))*cos(radians(lst)),cos(radians(lat))*sin(radians(lst)),sin(radians(lat)))*4.26e-5
    #vector subtraction to get NH to sun
    NHtosun = vector(ephem.position(JD, 10, 2)) - earthtoNH
    suntoasteroid = r
    #vector addition to get earth to asteroid
    earthtoasteroid = NHtosun + suntoasteroid
    #convert to unit vector and back to spherical coordinates
    earthtoasteroid /= mag(earthtoasteroid)
    dec = asin(earthtoasteroid.z)
    ra = atan(earthtoasteroid.y/earthtoasteroid.x)
    if earthtoasteroid.x < 0:
        ra += np.pi
    if ra < 0:
        ra += 2*np.pi
    return degrees(ra)/15., degrees(dec)

def rk4(r, rdot, ts):
	r1 = r
	rdot1 = rdot
	a1 = -(mu*r1/mag(r1)**3)

	r2 = r + 0.5*rdot1*ts
	rdot2 = rdot + 0.5*a1*ts
	a2 = -(mu*r2/mag(r2)**3)

	r3 = r + 0.5*rdot2*ts
	rdot3 = rdot + 0.5*a2*ts
	a3 = -(mu*r3/mag(r3)**3)

	r4 = r + rdot3*ts
	rdot4 = rdot + a3*ts
	a4 = -(mu*r4/mag(r4)**3)

	r_f = r + (ts/6.0)*(rdot1 + 2*rdot2 + 2*rdot3 + rdot4)
	rdot_f = rdot + (ts/6.0)*(a1 + 2*a2 + 2*a3 + a4)
	return r_f, rdot_f

def orbital_elements(r, r_dot):
    h = np.cross(r, r_dot)
    e = np.cross(r_dot, h) - r/mag(r)
    a = mag(h)**2/(1-mag(e)**2)
    q = a*(1-mag(e))
    i = acos(np.dot(h, vector(0.,0.,1.))/mag(h))
    N = np.cross(vector(0.,0.,1.), h)
    Omega = acos(N[0]/mag(N))
    if N[1] < 0:
        Omega = 2*np.pi - Omega
    omega = acos(np.dot(e,N)/(mag(e)*mag(N)))
    if e[2] < 0:
        omega = 2*np.pi - omega
    return a, mag(e), degrees(i), degrees(Omega), degrees(omega)

"""CONSTANTS"""
c = 173.145
k = 2*np.pi/365.25
mu = 1.0

#INITIAL TOTAL DATA SET
#all_RAs = [toDecimal(19.,29.,4.3), toDecimal(19.,25.,22.8), toDecimal(19.,21.,29.8), toDecimal(19.,17.,27.8), toDecimal(19.,13.,19.9), toDecimal(19.,9.,9.5)]
#all_Decs = [toDecimal(-4.,43.,34.), toDecimal(-4.,48.,7.), toDecimal(-4.,55.,34.), toDecimal(-5.,5.,59.), toDecimal(-5.,19.,25.), toDecimal(-5.,35.,52.)]
#all_times = [2457944.75, 2457947.75, 2457950.75, 2457953.75, 2457956.75, 2457959.75]

#PLOT ALL DATA
#plot(all_RAs, all_Decs)

"""METHOD OF GAUSS BEGINS HERE:
Initial 3 conditions
"""
#RAs = 15*np.array([all_RAs[0], all_RAs[2], all_RAs[5]])
#Decs = np.array([all_Decs[0], all_Decs[2], all_Decs[5]])
#times = np.array([all_times[0], all_times[2], all_times[5]])

RAs = 15*np.array([toDecimal(20,29,36.163),toDecimal(20,31,59.804),toDecimal(20,34,18.734)])
Decs = np.array([toDecimal(-13,15,35.69),toDecimal(-10,11,44.10),toDecimal(-6,51,26.98)])
times = np.array([2457950.73745, 2457954.74516, 2457959.09353])
"""
Initial 3 conditions
"""

def determineorbit(RAs, Decs, times):
    #rho hat vectors for each of the three measured dates
    rho1_hat = toRectangular(RAs[0], Decs[0])
    rho2_hat = toRectangular(RAs[1], Decs[1])
    rho3_hat = toRectangular(RAs[2], Decs[2])

    time1 = times[0]
    time2 = times[1]
    time3 = times[2]

    #tau times
    tau1 = k*(time1 - time2)
    tau3 = k*(time3 - time2)

    #sun-earth vectors
    R1 = ephem.position(times[0], 10, 2)
    R2 = ephem.position(times[1], 10, 2)
    R3 = ephem.position(times[2], 10, 2)

    #rho guess
    r2 = vector(1.5, 0., 0.)

    #scalar coefficients (a1, a3)
    a1 = g(tau3, r2)/(f(tau1, r2)*g(tau3, r2) - f(tau3, r2)*g(tau1, r2))
    a3 = -g(tau1, r2)/(f(tau1, r2)*g(tau3, r2) - f(tau3, r2)*g(tau1, r2))
    print a1, a3

    #converge on vector orbital elements
    i = 0
    while i < 10:

        #rho1 triple products
        rho1_1 = a1*np.dot(np.cross(R1, rho2_hat), rho3_hat)
        rho1_2 = np.dot(np.cross(R2, rho2_hat), rho3_hat)
        rho1_3 = a3*np.dot(np.cross(R3, rho2_hat), rho3_hat)
        #rho2 triple products
        rho2_1 = a1*np.dot(np.cross(rho1_hat, R1), rho3_hat)
        rho2_2 = np.dot(np.cross(rho1_hat, R2), rho3_hat)
        rho2_3 = a3*np.dot(np.cross(rho1_hat, R3), rho3_hat)
        #rho3 triple products
        rho3_1 = a1*np.dot(np.cross(rho2_hat, R1), rho1_hat)
        rho3_2 = np.dot(np.cross(rho2_hat, R2), rho1_hat)
        rho3_3 = a3*np.dot(np.cross(rho2_hat, R3), rho1_hat)
        #denom triple product
        denom = np.dot(np.cross(rho1_hat, rho2_hat), rho3_hat)

        #final rho's
        rho1 = (rho1_1-rho1_2+rho1_3)/(a1*denom)
        rho2 = (rho2_1-rho2_2+rho2_3)/(-denom)
        rho3 = (rho3_1-rho3_2+rho3_3)/(a3*denom)

        #final vector orbital elements
        r2 = rho2*rho2_hat - R2
        r1 = rho1*rho1_hat - R1
        r3 = rho3*rho3_hat - R3

        if i == 0:
            r2_dot = ((f(tau3,r2)/(g(tau1,r2)*f(tau3,r2)-g(tau3,r2)*f(tau1,r2)))*r1) - (f(tau1,r2)/(g(tau1,r2)*f(tau3,r2)-g(tau3,r2)*f(tau1,r2))*r3)
            a1 = g(tau3, r2)/(f(tau1, r2)*g(tau3, r2) - f(tau3, r2)*g(tau1, r2))
            a3 = -g(tau1, r2)/(f(tau1, r2)*g(tau3, r2) - f(tau3, r2)*g(tau1, r2))
            print a1, a3

        else:
            r2_dot = ((f(tau3,r2,r2_dot)/(g(tau1,r2,r2_dot)*f(tau3,r2,r2_dot)-g(tau3,r2,r2_dot)*f(tau1,r2,r2_dot)))*r1) \
                    -(f(tau1,r2,r2_dot)/(g(tau1,r2,r2_dot)*f(tau3,r2,r2_dot)-g(tau3,r2,r2_dot)*f(tau1,r2,r2_dot))*r3)
            a1 = g(tau3, r2,r2_dot)/(f(tau1, r2,r2_dot)*g(tau3, r2,r2_dot) - f(tau3, r2,r2_dot)*g(tau1, r2,r2_dot))
            a3 = -g(tau1, r2,r2_dot)/(f(tau1, r2,r2_dot)*g(tau3, r2,r2_dot) - f(tau3, r2,r2_dot)*g(tau1, r2,r2_dot))
            print a1, a3

        #new_time1 = time1 - rho1/c
        #new_time2 = time2 - rho2/c
        #new_time3 = time3 - rho3/c

        #R1 = ephem.position(new_time1, 10, 2)
        #R2 = ephem.position(new_time2, 10, 2)
        #R3 = ephem.position(new_time3, 10, 2)

        #tau1 = k*(new_time1-new_time2)
        #tau3 = k*(new_time3-new_time2)

        i += 1

    print r2, mag(r2), r2_dot

    """
    #rk4 integration
    time = all_times[4] - times[1]
    t = 0
    ts = 0.0001
    r_test, r_dot_test = r2, r2_dot
    while t < time*k:
        r_test, r_dot_test = rk4(r_test, r_dot_test, ts)
        print r_test, r_dot_test, mag(r_test)
        t += ts

    #generate ephemeris
    final_ra, final_dec = gen_ephem(41.3083, 72.9279, all_times[4], r_test)
    print toSexagesimal(final_ra), toSexagesimal(final_dec)
    """

    #generate classical orbital elements
    print orbital_elements(r2, r2_dot)

determineorbit(RAs, Decs, times)
