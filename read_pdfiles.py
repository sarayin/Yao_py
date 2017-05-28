"""
This module is an example code of how to read and parce a pandas datafile and do
basic calculation/manipulation with the data.
The data we use here is the Thacher Weather dataset that can be found in the hw1
file on https://gitlab.com/jonswift/DataSci
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

date_format = "%Y-%m-%d"

def month_info(month_ind):
    """
    Finds the length and name of a month given the month index (1,2,3...)
    """
    month_lengs = [31,28,31,30,31,30,31,31,30,31,30,31]
    month_names = ['January','February','March','April','May','June','July','August','September','October','November','December']
    return month_lengs[month_ind],month_names[month_ind]

def read_data(year,month=False,startday=False,endday=False,DFname='Temp Out'):
    """
    Function to return data of a specific dataframe from a specific time period,
    can also calculate the max and mins

    *args:
    year: int, e.g.2012

    **kwargs:
    month: int, e.g.4, default=False
    startday: int, e.g.12, default=False
    endday: int, e.g.30, default=False
    DFname: str, default='Temp Out'
    """
    years = np.array([2012,2013,2014,2015,2016])
    try:
        i, = np.where(years == year)[0]
    except:
        print 'No datafile for year '+str(year)
        return None

    print "Reading data from %s"%(year)
    if year < 2015:
        data = pd.read_table('/Users/sara/python/DataSci/ThacherWeather/WS_data_'+str(year)+'.txt',
                             usecols=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                             skiprows=[0,1],header=0,names=['Date','Time','Heat Index',
                                                             'Temp Out','Wind Chill','Hi Temp',
                                                             'Low Temp','Hum Out','Dew Pt.','Wind Speed',
                                                             'Wind Hi','Wind Dir','Rain','Barometer',
                                                             'Temp In','Hum In','Archive'],
                                                             na_values = ['--','---','------'])
    else:
        data = pd.read_table('/Users/sara/python/DataSci/ThacherWeather/WeatherLink_Data_'+str(year)+'.txt',
                             usecols=np.arrange(38),
                             skiprows=[0],header=0,names = ['Date', 'Time', 'Temp Out', 'Temp Hi', 'Temp Low',
                                                                 'Out Hum','Dew Pt','Wind Speed', 'Wind Dir','Wind Run',
                                                                 'Hi Speed','Hi Dir','Wind Chill','Heat Index','THW Index',
                                                                 'THWS Index','Barometer','Rain','Rain Rate','Solar Rad',
                                                                 'Solar Energy','Hi Rad','Solar Index','UV Dose','UV','Hi DD',
                                                                 'Heat DD','Cool Temp','In Hum','In Dew','In Heat','In EMC',
                                                                 'In Density','In Air ET','Samp','Wind Tx','Wind Recept',
                                                                 'ISS Int'],
                                                                 na_values = ['--','---','------'])
    datetimeDF = data['Date']+' '+data['Time'] #this is a data frame
    dtindex = pd.to_datetime(datetimeDF).values #datetime index values
    DFvalues = data[DFname].values #values of the column(dataframe)

    if month and startday and endday:
        year,month,startday,endday = str(year),str(month),str(startday),str(endday)
        startdate=year+'-'+month+'-'+startday
        enddate=year+'-'+month+'-'+endday
    elif month==True:
        year,month = str(year),str(month)
        startdate=year+'-'+month+'-1'
        enddate=year+'-'+month+'-'+str(month_length(month))
    else:
        year = str(year)
        startdate=year+'-1'+'-1'
        enddate=year+'-12'+'-31'
    dtDFI=pd.DataFrame(DFvalues,columns=[DFname],index=dtindex)
    result=dtDFI[startdate:enddate]

    days = datetime.strptime(startdate, date_format)-datetime.strptime(enddate, date_format)
    if len(result)!= days.days:
        print "Data incomplete!"
    return result


def calcMaxMin(year,month,DFname='Temp Out'):
    """
    Calculates maximum and minimum values in a column given year and month index

    *args:
    year: int, e.g.2012
    month: int, e.g.4

    **kwargs:
    DFname: str, default='Temp Out', the colomn you want to look at
    """
    maxs = []
    mins = []
    dtDFI = read_data(year,month=month,DFname=DFname)
    d = [31,28,31,30,31,30,31,31,30,31,30,31][int(month)-1]
    for i in np.arange(d)+1:
        date = str(year)+'-'+str(month)+'-'+str(i)
        try:
            vals = dtDFI[date].values.astype('float')
            maxs = np.append(maxs,np.nanmax(vals))
            mins = np.append(mins,np.nanmin(vals))
        except:
            pass
    if len(maxs)<=10:
        print "WARNING! Less than 10 maximum values"
    if len(mins)<=10:
        print "WARNING! Less than 10 minimum values"
    if len(maxs)!= 0 and len(mins)!=0:
        return maxs,mins
    else:
        return None

def compare_data(years=[2014,2015],month=4,DFname='Temp Out',binsize=5,normed=True):
    """
    Plot histograms of maximum and minimum values of the gien two years

    **kwargs:
    years: list, the two years of data that you want to compare
    month: int, the index of the month you want to compare
    DFname: str, default='Temp Out'
    binsize: int, default=5
    normed: boolean, default=True
    """
    try:
        maxs1, mins1 = calcMaxMin(years[0],month=month,DFname=DFname)
        maxs2, mins2 = calcMaxMin(years[1],month=month,DFname=DFname)

        plt.figure('Thacher Weather Station Compare Data',figsize=(8,10))
        plt.clf()
        plt.ion()

        plt.title('Temperature Highs and Lows of'+' '+month_info(month)[1]+str(years[0]))

        plt1 = plt.subplot(211)
        maxshist1 = plt1.hist(maxs1,alpha=0.5,bins=np.arange(45)*3,label=str(years[0]),color='b')
        maxshist2 = plt1.hist(maxs2,alpha=0.5,bins=np.arange(45)*3,label=str(years[1]),color='r')
        plt1.legend()
        plt1.set_xlabel('Outside Temperature ($^o$F)',fontsize=14)
        plt1.set_ylabel('Frequency',fontsize=14)
        plt1.set_title('High Temperature Distributions')


        plt2 = plt.subplot(212)
        minshist1 = plt2.hist(mins1,alpha=0.5,bins=np.arange(45)*3,label=str(years[0]),color='b')
        minshist2 = plt2.hist(mins2,alpha=0.5,bins=np.arange(45)*3,label=str(years[1]),color='r')
        plt2.legend()
        plt2.set_xlabel('Outside Temperature ($^o$F)',fontsize=14)
        plt2.set_ylabel('Frequency',fontsize=14)
        plt2.set_title('Low Temperature Distributions')

        plt.xlim(30,100)
        plt.show()
    except:
        print "No Values in given time period"

    return

"""
jswift comments and suggestions:
--------------------------------

No path specification for reading data. Must be in proper directory to run this routine.

Might want strategize your try and except statements a little better so that you can differentiate
between different failures.

I love the use of the special comments under the function definitions!

95/100
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_params import plot_params,plot_defaults

def get_files():
    """
    Function to return the years that weather data are available and
    the corresponding weather data text file
    """

    dpath = '/Users/jonswift/Thacher/Teaching/Classes/Data Science/Lessons'+\
    '/Explore Data/ThacherWeather/'

    years = np.array([2012,2013,2014,2015,2016])
    fnames = ['WS_data_2012.txt','WS_data_2013.txt','WS_data_2014.txt',\
             'WeatherLink_Data_2015.txt','WeatherLink_Data_2016.txt']
    files = np.array([dpath+f for f in fnames])
    return years, files


def read_year(year=2012):
    years,files = get_files()

    try:
        i, = np.where(years == year)[0]
    except:
        print 'No datafile for year '+str(year)
        return None

    na_values = ['--','---','------']

    if year < 2015:
        names = ['Date','Time','Heat Index',
                 'Temp Out','Wind Chill','Hi Temp',
                 'Low Temp','Hum Out','Dew Pt','Wind Speed',
                 'Wind Hi','Wind Dir','Rain','Barometer',
                 'Temp In','Hum In','Archive']
        cols = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        data = pd.read_table(files[i],usecols=cols,skiprows=[0,1],\
                             header=0,names=names,na_values=na_values)

    elif year >= 2015:
        names = ['Date', 'Time', 'Temp Out', 'Temp Hi', 'Temp Low', \
                 'Out Hum','Dew Pt','Wind Speed', 'Wind Dir','Wind Run',\
                 'Hi Speed','Hi Dir','Wind Chill','Heat Index','THW Index',\
                 'THWS Index','Barometer','Rain','Rain Rate','Solar Rad',\
                 'Solar Energy','Hi Rad','Solar Index','UV Dose','UV','Hi DD',\
                 'Heat DD','Cool Temp','In Hum','In Dew','In Heat','In EMC',\
                 'In Density','In Air ET','Samp','Wind Tx','Wind Recept',\
                 'ISS Int']
        cols = np.arange(38)
        data = pd.read_table(files[i],usecols=cols,skiprows=[0,1],\
                             header=0,names=names,na_values=na_values)

    return data


def get_values(column='Temp Out'):
    years,files = get_files()

    final_DF = pd.DataFrame()
    for i in range(len(years)):
        print 'Getting data for year '+str(years[i])
        data = read_year(years[i])
        col = data[column].values
        dtDF = data['Date']+' '+data['Time']
        dt = pd.to_datetime(dtDF).values
        dtDFI = pd.DataFrame(col,columns=[column],index=dt)
        if i != 0:
            final_DF = final_DF.append(dtDFI)

    return final_DF


def month_hi_lo(DF,month=1,year=2012):

    index = str(year)+'-'+str(month)

    his = []
    los = []
    for i in np.arange(31)+1:
        date = index+'-'+str(i)
        try:
            vals = DF[date].values.astype('float')
            his = np.append(his,np.max(vals))
            los = np.append(los,np.min(vals))
        except:
            pass

    if len(his) < 10:
        print 'Warning: less than 10 values for high temps'
    if len(los) < 10:
        print 'Warning: less than 10 values for low temps'

    return his,los


def compare_month_data(DF, month=1,years=[2014,2015],binsize=5,normed=True):

    his1,los1 = month_hi_lo(DF,month=month,year=years[0])
    his2,los2 = month_hi_lo(DF,month=month,year=years[1])

    date1 = str(years[0])+'-'+str(month)
    date2 = str(years[1])+'-'+str(month)

    plot_params()
    plt.ion()
    plt.figure(2,figsize=(8,10))
    plt.clf()
    ax1 = plt.subplot(211)
    maxhist1 = ax1.hist(his1,alpha=0.75,bins=np.arange(150)*binsize, \
                       label=date1,color='steelblue',normed=normed)
    maxhist2 = ax1.hist(his2,alpha=0.75,bins=np.arange(150)*binsize, \
                       label=date2,color='goldenrod',normed=normed)
    alldata = np.append(his1,his2)
    ax1.set_xlim(np.min(alldata)*0.8,np.max(alldata)*1.2)
    ax1.set_xlabel('Outside Temperature ($^\circ$F)',fontsize=14)
    ax1.set_ylabel('Frequency',fontsize=14)
    ax1.set_title('High Temperature Distributions')
    ax1.legend()
    ax2 = plt.subplot(212)
    minhist1 = ax2.hist(los1,alpha=0.75,bins=np.arange(150)*binsize, \
                        label=date1,color='steelblue',normed=normed)
    minhist2 = ax2.hist(los2,alpha=0.75,bins=np.arange(150)*binsize, \
                        label=date2,color='goldenrod',normed=normed)
    alldata = np.append(los1,los2)
    ax2.set_xlim(np.min(alldata)*0.8,np.max(alldata)*1.2)
    ax2.set_xlabel('Outside Temperature ($^\circ$F)',fontsize=14)
    ax2.set_ylabel('Frequency',fontsize=14)
    ax2.set_title('Low Temperature Distributions')
    ax2.legend()
    plt.tight_layout()

    plot_defaults()
    plt.ioff()
    return his1, his2, los1, los2


#DF = get_values()
#compare_month_data(DF,month=7,years=[2014,2016])
