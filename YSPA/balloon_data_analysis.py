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
balloon_file = [l.split(',') for l in balloon_file] # split rows into several columns
balloon_data = pd.DataFrame(balloon_file,columns=['ms','roll','pitch','heading','altitude','temp','xa','ya','za',],dtype = float)
geiger_file = map(lambda i: lines[i],np.arange(5,len(lines),3))
geiger_file = [l.split(',') for l in geiger_file]
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
