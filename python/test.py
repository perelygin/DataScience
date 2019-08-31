#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:11:40 2019

@author: perelygin
"""
def func1(i,j):
    return 1 if i==j else 0
    
z = [[func1(x,y) for x in range(5)] for y in range(5)]
print(z)