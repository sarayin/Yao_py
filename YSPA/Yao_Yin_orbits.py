#Also code for Problem set #6
from visual import *
import numpy as np
from math import *
from ephemPy import Ephemeris as Ephemeris_BC
from ephempy_example import *
import yao_master_script
import pandas as pd
import matplotlib.pyplot as plt
#from tqdm import tqdm

k = 2*pi/365. #modified day per day
#r, rdot = vector(0.244,2.17,-0.445),vector(-0.731,-0.0041,0.0502)
e = np.radians(23.43713) #tilt of the ecliptic
c = 173.145 #AU/day
Rearth = 4.26349651E-5
ephem = Ephemeris('405')

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
def get_orbit(r,rdot,delta_t = 0.01*k,mu = 1.,end_t=5.*k,vpython=False):
    t = 0
    if vpython:
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
        return obj.pos, vel
    else:
        while abs(t-end_t) > 1.00E-6:
            step = delta_t
            if t < end_t and t > t + 4.9*k:
                step = delta_t/10000.
            r, rdot = rk4(r,rdot,step)
            t+= step
            #print t , t/k
        return r, rdot
    #print obj.pos, vel, t

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

def get_ephemeris(JD, r, rdot, delta_t = 0.01*k, mu = 1.,verbose=True,\
                time_span =0.,lat = 41.3083, lon = -72.9279, equatorial = True):
    t = 0
    if time_span <0:
        delta_t = -delta_t
    while abs(time_span*k-t) > 1.00E-6:
        step = delta_t
        if t < time_span*k and t > (time_span-1.)*k:
            step = delta_t/100.
        r, rdot = rk4(r,rdot,step)
        t+= step
        #print t , t/k
    final_time = JD + time_span
    Rg = vector(ephem.position(final_time, 10, 2)) #vector that points from earth to sun, in equatorial coordinates
    g = vector(cos(lon)*cos(lat),sin(lon)*cos(lat),sin(lat))*Rearth #correct for parallex
    Rt = Rg - g
    if not equatorial:
        r = rotate(r, e, vector(1,0,0)) #rotate r to equatorial coordinates
    rho = Rg + r  # original rho
    """
    print rho,t,
    while t - (t-mag(rho)/c) > 0:
        r,rdot = rk4(r, rdot, -delta_t)
        t -= delta_t
    rho = Rg + r  #corrected rho
    """
    rho_hat = rho/mag(rho) #unit vector
    dec = np.arcsin(rho_hat.z) #calculate RA and Dec
    ra = np.arctan(rho_hat.y/rho_hat.x)
    if rho_hat.x < 0: #Check Quadrants
        ra += pi
    ra = (degrees(ra)/15.)
    if ra <= 0.:
        ra += 24.
    dec = degrees(dec)
    if verbose:
        print 'ra = ',yao_master_script.degrees_to_dms(ra) #15 degrees per hour
        print 'Dec = ',yao_master_script.degrees_to_dms(dec)
    return (ra, dec),(r,rdot),final_time

def f(rmag,tau):
    return 1-((tau**2)/(2*rmag**3))

def g(rmag,tau):
    return tau - (tau**3/(6*rmag**3))

def higher_f(r,rdot,rmag,tau):
    higher_deg = (tau**3*dot(r,rdot)/(2*rmag**5))+((tau**4)/24.)*(((3/rmag**3)*(dot(rdot,rdot)/rmag**2)-(1/rmag**3))-((15*dot(r,rdot)**2)/rmag**7)+(1./rmag**6))
    return f(rmag,tau) + higher_deg

def higher_g(r,rdot,rmag,tau):
    return g(rmag,tau) + tau**4*dot(r,rdot)/(4*rmag**5)

def rho(r,JD):
    vector(ephem.position(JD, 10, 2))
    r = rotate(r, e, vector(1,0,0))
    rho = Rg + r
    return rho

def rho_hat(ra,dec):
    ra,dec = np.radians(ra),np.radians(dec)
    return vector(cos(ra)*cos(dec),sin(ra)*cos(dec),sin(dec))

"""
asteroid_data = pd.read_table('2202_data.txt',sep = ',',names = ['time','ra','dec','mag'])
asteroid_data['julian_days'] = pd.DatetimeIndex(asteroid_data['time']).to_julian_date()
asteroid_data['ra'] = asteroid_data['ra'].apply(yao_master_script.dms_to_degrees)*15.
asteroid_data['dec'] = asteroid_data['dec'].apply(yao_master_script.dms_to_degrees)

#2a
plt.title('RA vs. Dec')
plt.plot(asteroid_data['ra'],asteroid_data['dec'],color = 'magenta',marker = '*',linestyle = 'None', markersize = 20)
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
plt.show()
"""
#Method of GAUSS
def method_of_GAUSS(RAs,Decs,JDs,index,rmag0 = 1.5):
    rho_hats = map(rho_hat,RAs,Decs)
    R_vectors = map(lambda d: vector(ephem.position(d,10,2)),JDs)
    rho_hat1,rho_hat2,rho_hat3 = [rho_hats[i] for i in index]
    R1, R2, R3 = [R_vectors[i] for i in index]
    Time1, Time2, Time3 = [JDs[i] for i in index]
    tau1 = k*(Time1-Time2)
    tau2 = 0.
    tau3 = k*(Time3-Time2)

    f1 = f(rmag0,tau1)
    g1 = g(rmag0,tau1)
    f3 = f(rmag0,tau3)
    g3 = g(rmag0,tau3)

    iteration = 0
    converge = False
    while converge == False :
        a1 = g3/(f1*g3-f3*g1)
        a3 = -g1/(f1*g3-f3*g1)
        #triple products
        tp1 = dot(cross(R1,rho_hat2),rho_hat3)
        tp2 = dot(cross(R2,rho_hat2),rho_hat3)
        tp3 = dot(cross(R3,rho_hat2),rho_hat3)

        tp4 = dot(cross(rho_hat1,R1),rho_hat3)
        tp5 = dot(cross(rho_hat1,R2),rho_hat3)
        tp6 = dot(cross(rho_hat1,R3),rho_hat3)

        tp7 = dot(cross(rho_hat2,R1),rho_hat1)
        tp8 = dot(cross(rho_hat2,R2),rho_hat1)
        tp9 = dot(cross(rho_hat2,R3),rho_hat1)

        denom = dot(cross(rho_hat1,rho_hat2),rho_hat3)

        rhomag1 = ((a1*tp1)-tp2+a3*tp3)/(a1*denom)
        rhomag2 = ((a1*tp4)-tp5+a3*tp6)/(-1*denom)
        rhomag3 = ((a1*tp7)-tp8+a3*tp9)/(a3*denom)

        #Time Correction
        Time_c1 = Time1 - (rhomag1/c)
        Time_c2 = Time2 - (rhomag2/c)
        Time_c3 = Time3 - (rhomag3/c)
        new_Times = [Time_c1,Time_c2,Time_c3]
        #print new_Times
        tau1 = k*(Time_c1-Time_c2)
        tau3 = k*(Time_c3-Time_c2)
        new_R_vectors = map(lambda d: vector(ephem.position(d,10,2)),new_Times)
        R1, R2, R3 = [new_R_vectors[i] for i in range(len(new_R_vectors))]
        rho1, rho2, rho3 = rho_hat1*rhomag1,rho_hat2*rhomag2,rho_hat3*rhomag3
        r1,r2,r3 = rho1-R1,rho2-R2,rho3-R3
        rdot2 = vector((f3 * r1)/(g1*f3-g3*f1) - (f1*r3)/(g1*f3-g3*f1))
        iteration +=1

        if abs(mag(r2)-rmag0) < 1E-6:
            converge = True
            print 'Converge successful!'
            print 'r2 = ',r2,'rdot2 = ',rdot2,'mag(r2) = ',mag(r2)#,'tau_c2 = ',tau_c2
            return r2, rdot2, mag(r2)#, tau_c2
        else:
            rmag0 = mag(r2)
            f1 = higher_f(r2,rdot2,rmag0,tau1)
            g1 = higher_g(r2,rdot2,rmag0,tau1)
            f3 = higher_f(r2,rdot2,rmag0,tau3)
            g3 = higher_g(r2,rdot2,rmag0,tau3)
            #print rmag0

def toDecimal(degrees, minutes, seconds):
    minutes = np.true_divide(minutes,60)
    seconds = np.true_divide(seconds,3600)
    if degrees < 0:
        return -(-degrees+minutes+seconds)
    return degrees+minutes+seconds

def rms(y,model):
    return np.sqrt(np.mean((model-y)**2))

marlough_data = pd.read_csv('marlough_data.csv',comment='#',names = ['JD','ra','dec','mag','lat','lon'])
marlough_data['ra'] = marlough_data['ra'].apply(yao_master_script.dms_to_degrees)*15.
marlough_data['dec'] = marlough_data['dec'].apply(yao_master_script.dms_to_degrees)

"""
#ra,dec,times = marlough_data['ra'],marlough_data['dec'],marlough_data['time']
RAs = 15*np.array([toDecimal(20,29,36.163),toDecimal(20,31,59.804),toDecimal(20,34,18.734)])
Decs = np.array([toDecimal(-13,15,35.69),toDecimal(-10,11,44.10),toDecimal(-6,51,26.98)])
times = np.array([2457950.73745, 2457954.74516, 2457959.09353])
"""
index = [0,18,36]
RAs = marlough_data['ra'].values
Decs = marlough_data['dec'].values
JDs = marlough_data['JD'].values
Lats = marlough_data['lat'].values
Lons = marlough_data['lon'].values
r_mid,rdot_mid,rmag_mid = method_of_GAUSS(RAs,Decs,JDs,index =index)
print r_mid, rdot_mid, rmag_mid
print get_orbital_elements(r_mid,rdot_mid)
ra,dec =  get_ephemeris(JDs[index[1]],r_mid,rdot_mid,lat = -30.1716, lon = -70.8009)[0]
#print get_ephemeris(times[3],r2,rdot2,time_span=3.)

# =====================
# Model Optimization
# =====================
real_coords = marlough_data[['ra','dec']].values
step_size=0.001
step_num = 100
rl = np.array([r_mid])
rdotl = np.array([rdot_mid])
predicted_coords = np.empty(shape = (0,2))
for d in range(len(marlough_data)):
    time_span = round(JDs[d]-JDs[index[1]],2)
    lat = Lats[d]
    lon = Lons[d]
    ra,dec = get_ephemeris(JDs[index[1]],r_mid,rdot_mid,time_span=time_span,verbose=False,lat = lat, lon = lon)[0]
    array = np.array([ra*15.,dec]) #convert to degrees
    predicted_coords = np.vstack((predicted_coords,array))
#ra_rms = rms(real_coords.T[0],predicted_coords.T[0])
#dec_rms = rms(real_coords.T[1],predicted_coords.T[1])
#new_rms = ra_rms+dec_rms
rms0 = rms(real_coords,predicted_coords)
rmsl = np.array([rms0])
print rms0
for i in range(step_num-1):
    accept_num = 0
    new_r = vector((rl[-1]+[np.random.normal(scale=step_size,size=3)])[0])
    new_rdot = vector((rdotl[-1]+[np.random.normal(scale=step_size,size=3)])[0])
    predicted_coords = np.empty(shape = (0,2))
    print 'new_r = ',new_r,'new_rdot = ',new_rdot
    for d in range(len(marlough_data)):
        time_span = round(JDs[d]-JDs[index[1]],2)
        lat = Lats[d]
        lon = Lons[d]
        ra,dec = get_ephemeris(JDs[index[1]],new_r,new_rdot,time_span=time_span,verbose=False,lat = lat, lon = lon)[0]
        array = np.array([ra*15.,dec]) #convert to degrees
        predicted_coords = np.vstack((predicted_coords,array))
    #ra_rms = rms(real_coords.T[0],predicted_coords.T[0])
    #dec_rms = rms(real_coords.T[1],predicted_coords.T[1])
    #new_rms = ra_rms+dec_rms
    new_rms = rms(real_coords,predicted_coords)
    print new_rms
    accept =  new_rms < rmsl[-1]
    if accept:
        accept_num +=1
    elif not accept:
        new_r,new_rdot, new_rms = rl[-1],rdotl[-1],rmsl[-1]
    rl,rdotl,rmsl = np.vstack((rl,new_r)),np.vstack((rl,new_rdot)),np.vstack((rmsl,new_rms))


#new_r =  <0.630579, -1.03445, -0.405584> new_rdot =  <0.88403, 0.46257, 0.399737>
#0.237245492533
