#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:13:05 2019

@author: perelygin
нулевая гипотеза
"""

#среднее значение всех выборочных средних равно среднему исходной совокупности
def mean(sample): # выборочное среднее
    return sum(sample)/len(sample)
    

#def sem(): #standard error of the mean SEM -стандартная ошибка среднего
    

smpl_mean = [4.51, 6.28]  # выборочные средние выборок
stnd_dev = [1.98,2.54] #standard deviation -  стандартное отклонение в выборках
n = 36 #число значений в выборке
m = 2 #число выборок

print(mean(smpl_mean))

