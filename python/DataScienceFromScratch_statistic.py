#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 19:22:01 2019

@author: perelygin
"""
import random 
from collections import Counter
import matplotlib.pyplot as plt # подмодуль pyplot

num_friends = [random.randrange(1,100,3) for j in range(250)] #количество друзей
friends_counts = Counter(num_friends) #Counter - ключи дают частотность(как часто встречается в списке) 
xs = range(101)
ys =  [friends_counts[x] for x in xs]
plt.bar(xs,ys)
plt.axis([0,120,0,20])
plt.title("Гистограма количество друзей")
plt.xlabel("Количество друзей")
plt.ylabel("Количество людей")
plt.show()
#print(friends_counts)
#print(ys)
#print(friends_counts[1])
num_points =  len(num_friends)
print('число точек "len(num_friends)" = ', num_points)

largest_value = max(num_friends)
print('максимальное значение "max(num_friends)"', largest_value)

smallest_value = min(num_friends)
print('минимальное значение "min(num_friends)"', smallest_value)

sorted_value = sorted(num_friends)
#print('Отсортированные значения',sorted_value)

smallest_value = sorted_value[0]
print('минимальное значение ', smallest_value)
second_smallest_value = sorted_value[1]
print('следующее минимальное значение ', second_smallest_value)

second_largest_value = sorted_value[-1]
print('следующий максимум ', second_largest_value)

def mean(x):
    """Среднее"""
    return sum(x)/len(x)

print('среднее ',mean(num_friends))

def median(v):
    """ возвращает ближайшее к середине значение для v"""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2  #индекс серединного значения
    if n % 2 == 1: # если нечетное, то вернуть срединное значение
        return sorted_v[midpoint]
    else: # если четное, то вернуть среднее значение 2х значений из середины
        lo = midpoint-1
        hi = midpoint
        return (sorted_v[lo]+sorted_v[hi]) / 2

print('медиана ',median(num_friends))
#print((sorted(num_friends)[124]+sorted(num_friends)[125])/2)

def quantile(x,p):
    """возвращает значение в x, соответствующее p-му проценту данных """
    p_index = int(p*len(x))   
    return sorted(x)[p_index]

print('Верхний квантиль ',quantile(num_friends,0.75))

























