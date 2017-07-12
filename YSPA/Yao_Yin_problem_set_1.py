import numpy as np
import math
#from tqdm import tqdm
print 'Problem Set 1 by Yao Yin'
#============Problem 1==============#
def calculate_pi(num):
    odds = np.array(range(1,num*2,4))
    evens = np.array(range(3,num*2,4))
    return sum(1./odds)-sum(1./evens)
print '=======Problem 1======='
print 'Testing the function...'
print 'The real pi/4 = ', math.pi/4.
print 'Approximation with 10 terms = ', calculate_pi(10)
print 'Approximation with 100 terms = ', calculate_pi(100)
print 'Approximation with 1000 terms = ', calculate_pi(1000)

#============Problem 2==============#

def bubble_sort(array, verbose = False):
    sorted = False
    steps = 0
    while not sorted:
        sorted = True
        for i in range(len(array)-1):
            if array[i] > array[i+1]:
                sorted = False
                array[i], array[i+1] = array[i+1], array[i]
                if verbose: print array
        steps += 1
    return array,steps

def comb_sort(array,scale=1.3, verbose = False):
    gap =  len(array)
    steps = 0
    sorted = False
    while not sorted :
        if gap == 1:
            sorted = True
        for i in range(len(array)-gap):
            if array[i] > array[i+gap]:
                array[i], array[i+gap] = array[i+gap], array[i]
                sorted = False
                if verbose: print array
        steps = steps +1
        if gap > 1:
            gap = max(1,int(gap/scale))
    return array,steps
print '=======Problem 2======='
print 'Testing the function...'
print 'Generating 1000 lists of 10 integers...'
from timeit import default_timer as timer
lists = []
bubble_times = []
comb_times = []
bubble_steps = []
comb_steps = []
sorted_lists = []
for i in range(1000):
    l = np.random.randint(low = -100, high = 100, size = 10)
    start = timer()
    bubble_steps.append(bubble_sort(l)[1])
    end = timer()
    bubble_times.append(end - start)
for i in range(1000):
    l = np.random.randint(low = -100, high = 100, size = 10)
    start = timer()
    comb_steps.append(comb_sort(l)[1])
    end = timer()
    comb_times.append(end - start)
print 'Average steps and time consumption using the comb sort method : %2f steps, %2f sec ' %(np.mean(comb_steps), np.mean(comb_times))
print 'Average steps and time consumption using the bubble sort method = %2f steps, %2f sec ' %(np.mean(bubble_steps), np.mean(bubble_times))
#print comb_times.mean(), bubble_times.mean()
#============Problem 3==============#
def degrees_to_dms(angle):
    degrees = int(angle)
    arcmin = int((angle - int(angle))*60.0)
    arcsec = (((angle - int(angle))*60.0) - arcmin)*60.0
    return '%.d:%.d:%.2f' % (degrees,arcmin,arcsec)

def dms_to_degrees(degrees, arcmin, arcsec):
     if degrees < 0.0:
          return degrees - (arcmin/60.0) - (arcsec/3600.0)
     else:
          return degrees + (arcmin/60.0) + (arcsec/3600.0)
print '=======Problem 3======='
print 'Testing the function...'
print dms_to_degrees(11,54,00)
print dms_to_degrees(-60,31,10)
print dms_to_degrees(-8,45,15.94)
print degrees_to_dms(60.04)
print degrees_to_dms(89.99999)
print degrees_to_dms(-23.43715)
degrees_to_dms(dms_to_degrees(90,0,0)-dms_to_degrees(41,18,58.8)+dms_to_degrees(-23,30,0))
degrees_to_dms(dms_to_degrees(18,29,16)-dms_to_degrees(20,26,50.13))
#============Problem 4==============#

def two_by_two(m):
    return m[0,0]*m[1,1]-m[0,1]*m[1,0]
def three_by_three(m):
    return m[0,0]*two_by_two(m[1:3,1:3])-m[0,1]*two_by_two(m[1:3,0:3:2])+m[0,2]*two_by_two(m[1:3,0:2])
print '=======Problem 4======='
print 'Testing the function...'
print 'Creating random matrix...'
m = np.random.rand(3,3)
print 'Matrix =', m
print 'Determinant calculated by numpy =', np.linalg.det(m)
print 'Determinant calculated by my function =',three_by_three(m)
#----------Extra Credit-------------#
def det(m):
    print m
    determinant = 0
    for i in range(len(m)):
        M = np.delete(m,i,0)
        for j in range(len(m)):
            #if i-1>2 and j-1>2:
                M = np.delete(M,j,1)
                print i,j,M
                determinant += (-1)**(i+j) * m[i,j] * det(M)
                print determinant
    return determinant
def sliceoutcolumn(inarray,column):
    front = inarray[:,0:column]
    back = inarray[:,column+1:]
    newarray = np.append(front,back,axis = 1)
    return newarray

#============Problem 5==============#

from visual import *
ball = sphere(pos=(-5,0,0), radius=0.5, color=color.cyan)
wallR = box(pos = (6,0,0), size=(0.2, 12, 12), opacity=0.5,
color=color.green)
wallL = box(pos = (-6, 0, 0), size = (0.2, 12, 12), opacity=0.5,
color=color.green)
wallB = box(pos = (0, -6, 0), size = (12, 0.2, 12), opacity=0.5, color
= color.green)
ball.trail = curve(color=ball.color)
ball.velocity = vector(10, 0, 0)
ball.accel = vector(0, -10, 0)
deltat = 0.005
t = 0.0
while t < 10:
     rate(100)
     ball.pos = ball.pos + ball.velocity*deltat
     ball.trail.append(pos=ball.pos)
     ball.velocity = ball.velocity + ball.accel*deltat
     if ball.pos.x > wallR.pos.x:
          ball.velocity.x = -1.0*ball.velocity.x
     if ball.pos.x < wallL.pos.x:
          ball.velocity.x = -1.0*ball.velocity.x
     if ball.pos.y < wallB.pos.y:
          ball.velocity.y = -1.0*ball.velocity.y
     t = t + deltat
