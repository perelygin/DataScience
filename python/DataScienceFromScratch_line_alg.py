#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:00:14 2019

@author: perelygin
"""
from functools import reduce 

def vector_add(v,w): #складывание векторов
    return[v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v,w): #вычитание векторов
    return[v_i - w_i for v_i, w_i in zip(v,w)]
    
def vector_sum(vectors) :
    result = vectors[0]
    for vect in vectors[1:]:
        result = vector_add(result,vect)
    return result

def vector_sum1(vectors):        
    return reduce(vector_add,vectors)
    
def scalar_mult(c,v):
    return[c*v_i for v_i in v]
    
    
    
v = [1,3,9]
w = [2,1,4]
e = [3,2,5]
print(vector_add(v,w))
print(vector_subtract(v,w))
l=[v,w,e]
print(vector_sum(l))
print(vector_sum1(l))
print(scalar_mult(10,v))