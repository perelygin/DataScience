
# coding: utf-8

# In[1]:



# gradient_descent.py

import sys
sys.path.append("../code-python3-ru")

from collections import Counter
from lib.linear_algebra import distance, vector_subtract, scalar_multiply
from functools import reduce
import math, random

def sum_of_squares(v):
    """вычисляет сумму квадратов элементов вектора v"""
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def plot_estimated_derivative():

    def square(x):
        return x * x

    def derivative(x):
        return 2 * x

    derivative_estimate = lambda x: difference_quotient(square, x, h=0.00001)

    # построить диаграмму, чтобы показать, что фактические производные
    # и их приближения в сущности одинаковые
    import matplotlib.pyplot as plt
    x = range(-10,10)
    plt.plot(x, map(derivative, x), 'rx')           # красный  x
    plt.plot(x, map(derivative_estimate, x), 'b+')  # синий +
    plt.show()                                      # фиолетовый *, будем надеяться

def partial_difference_quotient(f, v, i, h):

    # прибавить h только к i-му элементу v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]

def step(v, direction, step_size):
    """двигаться с шаговым размером step_size в направлении от v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

def safe(f):
    """определить новую функцию-обертку для f и вернуть ее"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')    # в Python так обоозначается бесконечность
    return safe_f

#
#
# минимизация / максимизация на основе пакетного градиентного спуска
#
#

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """использует градиентный спуск для нахождения вектора theta, 
    который минимизирует целевую функцию target_fn"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                           # установить тэта в начальное значение
    target_fn = safe(target_fn)               # безопасная версия целевой функции target_fn
    value = target_fn(theta)                  # минимизируемое значение

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # выбрать то, которое минимизирует функцию ошибок
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # остановиться, если функция сходится 
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    """вернуть функцию, которая для любого входящего x возвращает -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """то же самое, когда f возвращает список чисел"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)

#
# минимизация / максимизация на основе стохастического градиентного спуска 
#

def in_random_order(data):
    """генератор, который возвращает элементы данных в случайном порядке"""
    indexes = [i for i, _ in enumerate(data)]  # создать список индексов
    random.shuffle(indexes)                    # перемешать данные и
    for i in indexes:                          # вернуть в этом порядке
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = list(zip(x, y))
    theta = theta_0                             # первоначальная гипотеза
    alpha = alpha_0                             # первоначальный размер шага
    min_theta, min_value = None, float("inf")   # минимум на этот момент
    iterations_with_no_improvement = 0

    # остановиться, если достигли 100 итераций без улучшений
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # если найден новый минимум, то запомнить его
            # и вернуться к первоначальному размеру шага
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # в противном случае улучшений нет,
            # поэтому пытаемся сжать размер шага
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # и делаем шаг градиента для каждой из точек данных
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)

if __name__ == "__main__":

    print("применение градиента")

    v = [random.randint(-10,10) for i in range(3)]

    tolerance = 0.0000001

    while True:
        #print v, sum_of_squares(v)
        gradient = sum_of_squares_gradient(v)   # вычислить градиент в точке v
        next_v = step(v, gradient, -0.01)       # сделать шаг антиградиента
        if distance(next_v, v) < tolerance:     # остановиться, если сходимся
            break
        v = next_v                              # продолжить, если нет

    print("минимум v", v)
    print("минимальное значение", sum_of_squares(v))
    print()


    print("применение пакетной минимизации minimize_batch")

    v = [random.randint(-10,10) for i in range(3)]

    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)

    print("минимум v", v)
    print("минимальное значение", sum_of_squares(v))


# In[ ]:



