#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:19:46 2019

@author: perelygin
"""
from matplotlib import pyplot as plt

def make_chart_simple_bar_chart():
    movies     = ["Энни Холл", "Бен-Гур", "Касабланка", "Ганди", "Вестсайдская история"]
    num_oscars = [5, 11, 3, 8, 10]

    # ширина столбцов по умолчанию 0.8, поэтому добавим 0.1 к левым
    # координатам, чтобы каждый столбец был по центру интервала
    xs = [i + 0.1 for i, _ in enumerate(movies)]

    # построить столбцы с левыми X-координатами [xs] и высотой [num_oscars]
    plt.bar(xs, num_oscars)
    plt.ylabel("Количество наград", fontsize=11)
    plt.title("Мои любимые фильмы", fontsize=13)

    # добавить метки на оси X с названиями фильмов в центре каждого интервала
    plt.xticks([i + 0.5 for i, _ in enumerate(movies)], movies)
    plt.show()



years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]              # годы
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]  # ВВП

# создать линейную диаграмму:, годы по оси X, ВВП по оси Y
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

    # добавить название диаграммы
plt.title("Номинальный ВВП", fontsize=13)

    # добавить подпись к оси Y
plt.ylabel("Млрд $", fontsize=11)
plt.show()

make_chart_simple_bar_chart()
