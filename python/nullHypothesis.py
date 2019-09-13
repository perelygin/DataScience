#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:13:05 2019

@author: perelygin
нулевая гипотеза
дисперсионный анализ
"""
import math

def mean(sample): # выборочное среднее
    return round(sum(sample)/len(sample),2)
    
  
def sem(smpl_mean): #standard error of the mean SEM -стандартная ошибка среднего. Оно же стандартное отклонение по выборочным средним
# где smpl_mean - выборочные средние выборок  
    X_mean = mean(smpl_mean) #среднее значение всех выборочных средних равно среднему исходной совокупности
    R1 = [s_m - X_mean for s_m in smpl_mean]
    R2 = [(s_m - X_mean)**2 for s_m in smpl_mean]
    R3 = sum(R2)/(len(smpl_mean)-1)
    #print(R1,R2,sum(R2),len(smpl_mean))
    return round(math.sqrt(R3),2)

    
    

#smpl_mean = [66.9,73.2]  # выборочные средние выборок
#stnd_dev = [12.2, 14.4] #standard deviation -  стандартное отклонение в выборках
#n =61
#m = 2
    
#smpl_mean = [11.5,10.1,9.1]  # выборочные средние выборок
#stnd_dev = [1.3,2.1,2.4] #standard deviation -  стандартное отклонение в выборках
#n = 26
#m = 3

#smpl_mean = [4.51, 6.28]  # выборочные средние выборок
#stnd_dev = [1.98,2.54] #standard deviation -  стандартное отклонение в выборках
#n = 36 #число значений в выборке
#m = 2 #число выборок

#smpl_mean = [8.5,13.9]  # выборочные средние выборок
#stnd_dev = [4.7,4.1] #standard deviation -  стандартное отклонение в выборках
#n = 21
#m = 2

#smpl_mean = [3.17,2.72]  # выборочные средние выборок
#stnd_dev = [0.74,0.71] #standard deviation -  стандартное отклонение в выборках
#n = 200
#m = 2

smpl_mean = [85.1, 83.5, 80.9, 72.6, 60, 73.5, 63.8]  # выборочные средние выборок
stnd_dev =  [0.3,  1.0,  0.6,  0.7,  1.3, 0.7,  2.6] #standard deviation -  стандартное отклонение в выборках
n = 36
m = 7


s_in = n*(sem(smpl_mean)**2); #межгрупповая дисперсия
s_out = round(mean([sd**2 for sd in stnd_dev]),2);  #внутригрупповая дисперсия
v_in = m-1
v_out = m*(n-1)
print('Среднее по совокупности', mean(smpl_mean))
print('Стандартная ошибка среднего: ',sem(smpl_mean))
print('Межгрупповая дисперсия: ',s_in)
print('Внутригрупповая дисперсия: ',s_out)
print('F = s_меж/s_внут :', round(s_in/s_out,2))
print('Межгрупповое число степеней свободы',v_in)
print('Внутригрупповая число степеней свободы',v_out)