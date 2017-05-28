"""
This is an example code of how to do basic if else statements and basic manipulations
to pandas data frames.
This code analysis the length_data.csv file that can be found in the HW2 file on
https://gitlab.com/jonswift/DataSci
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:13:58 2016

@author: sara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('length_data.csv')

data[data['Name']== 'Yao']


plt.figure('Length Analysis')
plt.clf()
plt.ion()

plt.hist(data['Length'].values,bins=np.linspace(4.75,5.75,30),facecolor='white')

for name in data['Name'].unique():
    plt.hist(data[data['Name']==name]['Length'].values,bins=np.linspace(4.75,5.75,30),
             label=name, alpha=0.5,histtype='barstacked',stacked=True)

plt.legend(loc='upper left')

mean = np.std(data['Length'].values)
lengths = data[data['Length']>mean]['Length'].values
std = np.std(lengths)

# js comments
# great work, yao!
# 20/20
