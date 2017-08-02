
#====================================================#
# Git
#====================================================#
# Pull without entering message:
git pull --no-edit
# removes the file from the repo but also deletes it from the local file system.
git rm file.txt
#To remove the file from the repo and not delete it from the local file system use:
git rm --cached file.txt
#====================================================#
# Fitting
#====================================================#
# Fitting curves
from scipy.optimize import curve_fit
def func(X,ita,alpha):
    return ita * np.exp(-alpha*X)
params = curve_fit( func , X , E/AS )

#====================================================#
# Datetime
#====================================================#
#------Convert list to datetime objects
datetime(*map(int, start)).strftime('%Y-%m-%dT%H:%M:%S')
import datetime
values = ['2014', '08', '17', '18', '01', '05']
datetime.datetime(*map(int, values))
datetime.datetime(2014, 8, 17, 18, 1, 5)
#-------Loop through dates:
def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta
#--------Date From String:
stime = datetime.strptime(start,"%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%dT%H:%M:%S')
# Between Specific times:
masked_DF.between_time('11:00','13:00')
#Complex formatting:
datetime.strptime("December 12, 2012 at 07:07AM",'%B %d, %Y at %H:%M%p')
#====================================================#
# PANDAS
#====================================================#
#Loop through dates:
import pandas as pd
daterange = pd.date_range(start_date, end_date,freq='H',tz='UCT')
dates = pd.date_range('2016-9-11 00:00:00','2017-9-11 00:00:00',freq='H')
dates + timedelta(hours =7)

#-------Pandas indexing
# search with conditions
masked_DF.loc[masked_DF[60]<0]
# change values under certain conditions
cloud_DF.ix[cloud_DF['SkyTemp']==998.,'SkyTemp'] = np.nan
#-------Append row to Pandas DataFrame
mycolumns = ['A', 'B']
df = pd.DataFrame(columns=mycolumns)
rows = [[1,2],[3,4],[5,6]]
for row in rows:
    df.loc[len(df)] = row
#-------Plotting
# Histograms:
DF.plot.hist()
# Histograms with subplots:
DF.hist()
# Creating Subplots:
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10,5))
# Customize colors and stuff:
df.plot(color = 'lightgreen', marker = 'o')
style = 'ro' #don't use style and color at the same time

#-------Some data parsing examples
F = masked_DF.loc[masked_DF['W_avg']/masked_DF['power_output'] < 0.05]
days = pd.date_range(datetime(2016, 9, 1), datetime.now().date())
dailyDF = pd.DataFrame(columns = np.arange(61))
for d in pd.date_range(datetime(2016, 9, 1), datetime.now().date()):
    dailyDF.loc[len(dailyDF)] = DF[range(61)].ix[str(d.date())].sum().values
dailyDF = dailyDF.set_index(days)

#====================================================#
# Basics
#====================================================#
#--------Reload imported module
reload(my_module)
#-------Time your function
from timeit import default_timer as timer

start = timer()
# ...
end = timer()
print(end - start)
#Assigning variables using a loop and a dictionary:
elements = {}
for x in range(3):
    elements['rho_hat{0}'.format(x)]= rho_hats[index[x]]
    elements['R{0}'.format(x)] = R_vectors[index[x]]
# Unpack lists in calling functions:
my_list_of_variables = [a,b,c]
my_func(*my_list_of_variables) == my_func(a,b,c)
#====================================================#
# Matplotlib
#====================================================#
#Plotting lines with markers
plt.plot(range(10), '--bo')
#====================================================#
# Numpy
#====================================================#
To generate a random order of this set, you could use the random.choice function, e.g.:
from numpy import random
#starting list that needs to be sorted
b = list([5, 0, 9, 8, 7, 1, 4, 3, 2, 6])
a = random.choice(b, size=10, replace=False)
