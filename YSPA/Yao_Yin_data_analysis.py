import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from math import *
# ==============================
# Balloon Data Analysis
# ==============================
lines = open('balloon-data.txt').readlines()
balloon_file = map(lambda i: lines[i],np.arange(4,len(lines),3)) #Map function applies the function to all values in a list
geiger_file = map(lambda i: lines[i],np.arange(5,len(lines),3))
balloon_file = [l.split(',') for l in balloon_file] # split rows into several columns
geiger_file = [l.split(',') for l in geiger_file]
balloon_data = pd.DataFrame(balloon_file,columns=['ms','roll','pitch','heading','altitude','temp','xa','ya','za',],dtype = float)
geiger_data = pd.DataFrame(geiger_file,dtype = float)[[1,3,5]]
geiger_data.columns = ['CPS','CPM','uSv/hr']
geiger_data = geiger_data.convert_objects(convert_numeric=True) #Convert wierd str / object type objects to floats, returns NaN value for errors

#Millisecond vs altitude plot
ax1 = balloon_data.plot.scatter('ms','altitude')
ax1.set_ylabel('Altitude(meters)')
ax1.set_xlabel('Milliseconds')
ax1.set_title('Altitude of the balloon as a function of time')
#Temperature vs altitude plot
ax2 = balloon_data.plot.scatter('altitude','temp')
ax2.set_ylabel('Altitude(meters)')
ax2.set_xlabel('Temperature(celcius)')
ax2.set_title('Outside temperature as a function of altitude')
#Micro-Sieverts/hour vs altitude plot
plt.figure()
plt.scatter(balloon_data['altitude'],geiger_data['uSv/hr'])
plt.title('Micro-Sieverts/hour vs. Altitude')
plt.xlabel('Altitude(meters)')
plt.ylabel('Micro-Sieverts/hour')
#magnitude of acceleration vs time plot
plt.figure()
accl = np.sqrt(balloon_data['xa'].values**2+balloon_data['ya'].values**2+balloon_data['za'].values**2)
plt.scatter(balloon_data['ms'],accl)
plt.xlabel('Milliseconds')
plt.ylabel('Acceleration(m/s^2)')
plt.title('Magnitude of acceleration vs. time')
plt.show()

#millisec since start, roll, pitch, heading,
# altitude (m, based on pressure), temp (C), accel.x, accel.y, accel.z
# Geiger data
"""
weather_data = pd.read_csv('Condition_Sunrise-newhaven.csv',sep = ',', names = ['Date','Temp_avg','Sky','Temp_high','Temp_low'])
julian_days = pd.DatetimeIndex(weather_data['Date']).to_julian_date()
weather_data['julian_days'] = julian_days

weather_data['Skycode'] = np.zeros(len(weather_data))
sky_params = np.unique(weather_data['Sky'])
for d in range(len(sky_params)):
    weather_data.ix[weather_data['Sky']==sky_params[d],'Skycode'] = d
plt.figure(1,figsize = (10,10))
plt.title('Weather Conditions at Yale')
plt.plot(weather_data['julian_days'],weather_data['Skycode'],'.')
plt.yticks(range(len(sky_params)),sky_params)
plt.xlabel('Julian Day')
plt.tight_layout()
#Trial and error
lows = weather_data['Temp_low'].values
highs = weather_data['Temp_high'].values
def func(x,a,b,h,k):
    return a*np.sin(b*(x-h))+k
x = np.linspace(1,2000,20000)
y1 = func(x,35, 2*pi/360.,135 ,59)
y2 = func(x,36, 2*pi/360.,142,45)
#plt.plot(x,y1,label = 'High fit')
#plt.plot(x,y2,label = 'Low fit')

#Scipy fitting
from scipy.optimize import curve_fit
lows = np.nan_to_num(lows)
highs = np.nan_to_num(highs)
params1 = curve_fit(func,np.array(weather_data.index),highs, p0 = [35,2*pi/360.,135,59])[0]
params2 = curve_fit(func,np.array(weather_data.index),lows, p0 = [36, 2*pi/360., 142,45])[0]
plt.figure(2)
weather_data['Temp_high'].plot(label = 'Daily Highs',alpha = 0.5)
weather_data['Temp_low'].plot(label = 'Daily Lows', alpha = 0.5)
plt.plot(x,func(x,*params1), label = 'scipy highs')
plt.plot(x,func(x,*params2),label = 'scipy lows')
plt.legend()
plt.show()

#from astropy.io import ascii
#ascii.read(file)
"""
