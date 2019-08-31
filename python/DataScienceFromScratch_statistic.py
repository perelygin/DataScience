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
print('максимальное значение "max(num_friends)"', largest_value)