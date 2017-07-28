#Also code for Problem set #6
from visual import *
import numpy as np
from math import *
from ephemPy import Ephemeris as Ephemeris_BC
from ephempy_example import *
import yao_master_script
import pandas as pd

k = 2*pi/365. #modified day per day
r, rdot = vector(0.244,2.17,-0.445),vector(-0.731,-0.0041,0.0502)
e = np.radians(23.43713) #tilt of the ecliptic
c = 0.00200399 #AU/sec
Rearth = 4.26349651E-5

def get_orbital_elements(r, rdot, mu =1., verbose = True):
    h = cross(r, rdot)
    e = (cross(rdot, h)/mu)-(r/mag(r))
    SLR = mag(h)**2/mu #semi lactus rectum
    q = SLR / (1+mag(e)) #perihelion distance
    a = SLR / (1-mag(e)**2) #semi major axis
    i = degrees(np.arccos(h.z/(mag(h)))) # inclination angle
    N = cross(vector(0,0,1),h) # Ascending node
    OMEGA = degrees(np.arccos(N.x / mag(N)))
    if N.y < 0. :
        OMEGA = 180 - OMEGA
    omega = degrees(np.arccos(np.dot(e,N)/(mag(e)*mag(N))))
    if e.z < 0. :
        omega = 360 -omega
    if verbose:
        print 'Vector h = ',h
        print 'Vector e = ',e
        print 'Eccentricity (mag(e)) = ',mag(e)
        print 'Perihelion distance (q) = ',q
        print 'Semi major axis (a) = ',a
        print 'Inclination Angle (i) = ', i
        print 'Ascending Node = ', N
        print 'Longitude of Ascending Node (OMEGA) = ', OMEGA
        print 'Argument of Perihelion (omega) = ', omega
    return h, e, SLR, q, a, i, N, OMEGA, omega
def plot_orbit(r,rdot,delta_t = 0.01*k,mu = 1.,end_t=5.*k):
    t = 0
    h, e, SLR, q, a, i, N, OMEGA, omega = get_orbital_elements(r,rdot,verbose=False)
    xaixis = arrow(pos=(0,0,0),axis=(1,0,0),shaftwidth=0.01, length = 10, color = color.red)
    yaixis = arrow(pos=(0,0,0),axis=(0,1,0),shaftwidth=0.01, length = 10, color = color.green)
    zaixis = arrow(pos=(0,0,0),axis=(0,0,1),shaftwidth=0.01, length = 10,color = color.blue)
    ecliptic = cylinder(pos=(0,0,0), axis = (0,0,1), radius = 10, length =0.01,opacity=0.1)

    sun = sphere(pos = (0,0,0), radius = 0.5, color= color.yellow)
    obj = sphere(pos = r, radius = 0.1, color = color.red)
    vel = rdot
    trail = curve(color = color.red, retain = 50000)
    arrow(pos = (0,0,0), axis = h, shaftwidth = 0.02, length = mag(h), color = color.magenta)
    label(pos = h, text = 'h', color = color.magenta)
    arrow(pos = (0,0,0), axis = e, shaftwidth = 0.02, length = mag(h), color = color.cyan)
    label(pos = e, text = 'e', color = color.cyan)
    arrow(pos = (0,0,0), axis = N, shaftwidth = 0.02, length = mag(h), color = color.yellow)
    label(pos = N, text = 'N', color = color.yellow)
    while abs(t-end_t) > 1.00E-6:
        rate(1000)
        step = delta_t
        if t < end_t and t > t + 4.9*k:
            step = delta_t/10000.
        obj.pos, vel = rk4(obj.pos,vel,step)
        trail.append(pos=obj.pos,retain=50000)
        t+= step
        print t , t/k
    #print obj.pos, vel, t
    return obj.pos, vel
def a(vector): #Acceleration due to gravity
    return -vector/(mag(vector)**3)
#t = 0
def euler(r, rdot, delta_t):
    v_next = rdot + a(r)*delta_t
    r_next = r + v_next*delta_t
    return r_next,v_next
def rk4(r, rdot, delta_t):
    r1 = r
    v1 = rdot
    k1 = a(r1)

    v2 = v1 + k1*delta_t*0.5
    r2 = r1 + v1*delta_t*0.5
    k2 = a(r2)

    v3 = v1 + k2*delta_t*0.5
    r3 = r1 + v2*delta_t*0.5
    k3 = a(r3)

    v4 = v1 + k3*delta_t
    r4 = r1 + v3*delta_t
    k4 = a(r4)

    v_next = v1 + (k1+2*k2+2*k3+k4)/6.0 * delta_t #modified acceleration
    r_next = r1 + (v1+2*v2+2*v3+v4)/6.0 * delta_t
    return r_next, v_next
def main():
    xaixis = arrow(pos=(0,0,0),axis=(1,0,0),shaftwidth=0.03, length = 10, color = color.blue)
    yaixis = arrow(pos=(0,0,0),axis=(0,1,0),shaftwidth=0.03, length = 10, color = color.green)
    zaixis = arrow(pos=(0,0,0),axis=(0,0,1),shaftwidth=0.03, length = 10,color = color.orange)
    ecliptic = cylinder(pos=(0,0,0), axis = (0,0,1), radius = 10, length =0.01,opacity=0.1)

    sun = sphere(pos = (0,0,0), radius = 0.5, color= color.yellow)

    delta_t = 0.01

    rk4_obj = sphere(pos = (0.9,0,0), radius = 0.1, color = color.red)
    rk4_vel = vector(0,1.3,0) #velocity
    rk4_trail = curve(color = color.red)

    euler_obj = sphere(pos = (0.9,0,0), radius = 0.1, color = color.blue)
    euler_vel = vector(0,1.3,0)
    euler_trail = curve(color = color.blue)
    t = 0
    end_time = t + 5*k
    while abs(t-end_time) > 1.00E-12:
        rate(1000)
        rk4_obj.pos, rk4_vel = rk4(rk4_obj.pos,rk4_vel,delta_t)
        rk4_trail.append(pos=rk4_obj.pos,retain=50000)

        euler_obj.pos, euler_vel = euler(euler_obj.pos, euler_vel, delta_t)
        euler_trail.append(pos=euler_obj.pos,retain=50000)
        t += delta_t
        print rk4_obj.pos, rk4_vel, t
        """
        if t >= 8.09:
            print "The asteroid has finished 1/2 of its sidereal period!"
            print rk4_obj.pos, euler_obj.pos, euler_vel,t
            break

        if abs((mag(rk4_obj.pos)-mag(euler_obj.pos)))/mag(rk4_obj.pos) >= 0.01:
            print "The prediction of the two methods differs by more than 1%!"
            print rk4_obj.pos, euler_obj.pos, t
            break
        """

def get_ephemeris(JD, r, lat = 41.3083, lon = -72.9279):
    ephem = Ephemeris('405')
    Rg = vector(ephem.position(JD, 10, 2)) #vector that points from earth to sun, in equatorial coordinates
    """
    #correct for parallex
    g = vector(cos(lon)*cos(lat),sin(lon)*cos(lat),sin(lat))*Rearth
    Rt = Rg - g
    #correct for speed of light
    time_delta = mag(r)/c
    r += rdot * time_delta
    """
    r = rotate(r, e, vector(1,0,0)) #rotate r to equatorial coordinates
    #calculate rho
    roh = Rg + r
    roh_hat = roh/mag(roh) #unit vector
    #calculate RA and Dec
    dec = np.arcsin(roh_hat.z)
    ra = np.arctan(roh_hat.y/roh_hat.x)
    if roh_hat.x < 0:
        ra += pi
    #print degrees(ra), degrees(np.arccos(roh_hat.x/cos(dec)))
    print 'ra = ',yao_master_script.degrees_to_dms(degrees(ra)/15.) #15 degrees per hour
    #print 'RA = ', degrees_to_dms(degrees(np.arccos(roh_hat.x/cos(dec))/15.))
    print 'Dec = ',yao_master_script.degrees_to_dms(degrees(dec))
    print degrees(dec)
    return ra, dec, Rg, r, roh
#main()
#plot_orbit(r,rdot)
#JD on July1 , 2017 at 0:00 EDt is 2457935.666667
#et_ephemeris(2457940.666667,plot_orbit(r,rdot)[0])
#print get_ephemeris(2457940.666667,plot_orbit(r,rdot)[0])
def f(r,rdot,tau):
    deg_3 = 1-((tau**2)/(2*r**3))+((tau**3)*dot(r,rdot)/(2*r**5))
    deg_4 = (1/24.)*((3/r**3)*((dot(rdot,rdot))-(1/r**3))-((15*dot(r,rdot)**2)/r**7)+(1/r**6))*(tau**4)
    return deg_3 + deg_4

def g(r, rdot, tau):
    return tau - (tau**3/(6*r**3)) + ((r+rdot)*tau**4/(4*r**5))


asteroid_data = pd.read_table('2202_data.txt',sep = ',',names = ['time','ra','dec','mag'])
asteroid_data['julian_days'] = pd.DatetimeIndex(asteroid_data['time']).to_julian_date()
#asteroid_data['ra'],asteroid_data['dec'] = map(dms_to_degrees,asteroid_data['ra'])
