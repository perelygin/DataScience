
# coding: utf-8

# In[1]:



# linear_algebra.py

import re, math, random # регулярные выражения, математические функции и случайные числа
import matplotlib.pyplot as plt # подмодуль pyplot
from collections import defaultdict, Counter
from functools import partial, reduce

#
# функции для работы с векторами
#

def vector_add(v, w):
    """покомпонентное сложение двух векторов"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """покомпонентное вычитание двух векторов"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """вычислить вектор, чей i-й элемент - это среднее значение
    всех i-х элементов входящих векторов"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
    return math.sqrt(squared_distance(v, w))

#
# функции для работы с матрицами
#

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    """возвращает матрицу размером num_rows x num_cols,
    (i,j)-й элемент которой равен функции entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]

def is_diagonal(i, j):
    """единицы по диагонали, остальные нули"""
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)

# пользователь  0  1  2  3  4  5  6  7  8  9
#
friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # пользователь 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # пользователь 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # пользователь 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # пользователь 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # пользователь 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # пользователь 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # пользователь 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # пользователь 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # пользователь 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # пользователь 9

#####
# МОЖНО УДАЛИТЬ ВСЕ, ЧТО НИЖЕ
#


def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("нельзя складывать матрицы с разными формами")

    num_rows, num_cols = shape(A)
    def entry_fn(i, j): return A[i][j] + B[i][j]

    return make_matrix(num_rows, num_cols, entry_fn)


def make_graph_dot_product_as_vector_projection(plt):

    v = [2, 1]
    w = [math.sqrt(.25), math.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0,0]

    plt.arrow(0, 0, v[0], v[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0 ,0, w[0], w[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(v•w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1],
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v,w,o),marker='.')
    plt.axis('equal')
    plt.show()


# In[ ]:



