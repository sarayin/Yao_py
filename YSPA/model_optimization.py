import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from math import *

weather_data = pd.read_csv('Condition_Sunrise-newhaven.csv',sep = ',', names = ['Date','Temp_avg','Sky','Temp_high','Temp_low'])
julian_days = pd.DatetimeIndex(weather_data['Date']).to_julian_date()
weather_data['julian_days'] = julian_days
weather_data['Skycode'] = np.zeros(len(weather_data))
"""
sky_params = np.unique(weather_data['Sky'])
for d in range(len(sky_params)):
    weather_data.ix[weather_data['Sky']==sky_params[d],'Skycode'] = d
plt.figure(1,figsize = (10,10))
plt.title('Weather Conditions at Yale')
plt.plot(weather_data['julian_days'],weather_data['Skycode'],'.')
plt.yticks(range(len(sky_params)),sky_params)
plt.xlabel('Julian Day')
plt.tight_layout()

"""

"""
#Trial and error
lows = weather_data['Temp_low'].values
highs = weather_data['Temp_high'].values
x = np.linspace(1,2000,20000)
y1 = sinfunc(x,[35, 2*pi/360.,135 ,59])
y2 = sinfunc(x,[36, 2*pi/360.,142,45])
#plt.plot(x,y1,label = 'High fit')
#plt.plot(x,y2,label = 'Low fit')

def func(x,a,b,h,k):
    return a*np.sin(b*(x-h))+k
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
"""
def func(x,params):
    a,b,h,k = params
    return a*np.sin(b*(x-h))+k
def rms(y,model):
    return np.sqrt(np.mean((model-y)**2))


pred_high = [35,2*pi/360.,135,59]
pred_low = [36,2*pi/360.,135,59]

def gradient_climbing(pred,x,y,params=4,step_size=0.1,step_num = 10000,plot=True):
    accept_num = 0
    posl = np.array([pred])
    rmsl = np.array([rms(y,func(x,posl[-1].tolist()))])
    for i in range(step_num-1):
        new_pos = posl[-1]+np.random.normal(scale=step_size,size=params)
        new_rms = rms(y,func(x,new_pos.tolist()))
        accept =  new_rms < rmsl[-1]
        if accept:
            accept_num +=1
        elif not accept:
            new_pos, new_rms = posl[-1],rmsl[-1]
        posl,rmsl = np.vstack((posl,new_pos)),np.vstack((rmsl,new_rms))
    if plot:
        fig = plt.figure(figsize=(12,5),dpi=80)
        plt.xlabel('Step #')
        plt.title('Parameter Positions')
        for p in range(params):
            ax = fig.add_subplot(params,1,p+1)
            ax.set_ylabel('P'+str(p+1))
            ax.plot(range(step_num),posl[:,p])
        plt.legend()
        plt.show()
        #plot curve in function
    return posl, rmsl, accept_num

posl,rmsl,accept_num = gradient_climbing(pred_high,np.array(weather_data.index),weather_data['Temp_high'])
print posl[-1],rmsl[-1]

plt.figure()
#plt.plot(weather_data['julian_days'],weather_data['Temp_high'],'bo',label = 'Daily Highs',alpha = 0.5)
weather_data['Temp_high'].plot(label = 'Daily Highs',alpha = 0.5)
x = np.linspace(1,2000,20000)
y = func(x,posl[-1])
plt.plot(x,y)
plt.show()
#from astropy.io import ascii
#ascii.read(file)
