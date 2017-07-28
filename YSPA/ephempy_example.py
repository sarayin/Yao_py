#Code to use EphemPy to get Rsun from JPL binary ephemeris tables

from visual import *

from ephemPy import Ephemeris as Ephemeris_BC
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


def JD_to_lst(JDT, longi):
     longi = longi/15.
     #JD reference time
     JDT_ref = 2457388.5
     #LST reference time
     LST_ref = 6. + (40./60) + (21./3600)
     #Solar Days that have passes since reference time
     delta_JDT = JDT - JDT_ref
     #sidereal hours that have passed
     side_hours = 24. * (366.2422/365.2422) * delta_JDT
     #remove integer sidereal days
     add_time = side_hours%24
     return (LST_ref + longi + add_time)%24

ephem = Ephemeris('405')

print ephem.position(2457570.666667, 10, 2)

R_JPL = vector(-1.682888807766579E-01,  9.199634656377305E-01,  3.988105652922423E-01,)
print R_JPL
R_EMB = vector(-1.683058465580201E-01,  9.199402031694297E-01,  3.988031711799118E-01)

print R_EMB
'''

# New Haven longitude in degrees
longitude =-72.9279
latitude = 40.3083
eps = radians(-23.43731)

# radius of the earth in AU
Radius_earth = 4.26349651E-05

#July 1, 2016 at 4h UT
JD = 2457570.666667

# geocentric Sun vector
R_sun = vector(ephem.position(2457570.666667,  10, 2))
print "Sun vector: ", R_sun

# get geocentric vector for New Haven
lst = JD_to_lst(JD, longitude)
lst = lst * 15.
print 'LST at New Haven = ', lst
lst = radians(lst)
lat = radians(latitude)

g = vector(cos(lst)*cos(lat), sin(lst)*cos(lat), sin(lat))
g = g*Radius_earth
print 'geocentric vector in equatorial coordinates = ', g

g = 0.*g

R_topo = R_sun - g

# asteroid position vector in ecliptic coordinates
# 1866 Sisyphus

r_ast = vector(-1.456374007296408E+00, -1.527852430965919E+00,  5.441666739176260E-01)

print "asteroid position: ", r_ast
r_ast_equ = rotate(r_ast, -eps, vector(1,0,0))
print "asteroid position: ", r_ast_equ

rho = r_ast_equ + R_topo
print "range vector: ", rho

rho_hat = rho/mag(rho)
print "rho hat: ", rho_hat

DEC = asin(rho_hat.z)
print "DEC: ", degrees(DEC)

RA = acos(rho_hat.x/cos(DEC))
if rho_hat.y < 0:
    RA = 2.*pi - RA

print "RA: ", degrees(RA)

'''
