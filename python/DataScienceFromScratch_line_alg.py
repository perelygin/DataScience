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

def vector_sum1(vectors):  #через свертку списка      
    return reduce(vector_add,vectors)
    
def scalar_mult(c,v):
    return[c*v_i for v_i in v]
    
def dot(c,v): #Cкалароное произведение векторов - сумма покомпонентных произведений. 
              #Это длина вектора,  которая получится если спроецировать вектор v на w
    """v_1*c_1+v_n*c_n """
    return  sum(c_i*v_i for c_i, v_i in zip(c,v))
    
def  shape(A):#форма матрицы
    num_row = len(A)
    num_cols = len(A[0]) if A else 0  #число элементов в первой строке
    return num_row,num_cols
   
def get_row(A,i):
    return A[i]

def get_column(A,j):
    return [A_i[j] for A_i in A]


def make_matrix(rows,cols,entry_fn):
    return [[entry_fn(i,j)
             for j in range(cols)]
             for i in range(rows)]
   
def is_diagonal(i,j):
    return 1 if i==j else 0


v = [1,3,9]
w = [2,1,4]
e = [3,2,5]
print(vector_add(v,w))
print(vector_subtract(v,w))
l=[v,w,e]
print(vector_sum(l))
print(vector_sum1(l))
print(scalar_mult(10,v))
print(dot(w,v))

#матрицы
A = [[1,2,3],
     [4,5,6]]
B = [[1,2],
     [3,4],
     [5,6]]

print(shape(B))
print(get_row(A,0))
print(get_column(A,2))
print(make_matrix(5,5,is_diagonal))


