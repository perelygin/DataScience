
# coding: utf-8

# In[1]:



# probability.py

from collections import Counter
import math, random

def random_kid():  # произвольно выбрать мальчика или девочку
    return random.choice(["boy", "girl"])

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    """возвращает вероятность того, что равномерно распределенная
    случайная величина <= x"""
    if x < 0:   return 0    # величина никогда не бывает меньше 0
    elif x < 1: return x    # например, P(X <= 0.4) = 0.4
    else:       return 1    # величина всегда меньше 1

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def plot_normal_pdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend()
    plt.show()

def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def plot_normal_cdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend(loc=4) # снизу справа
    plt.show()

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """найти приближенную инверсию, используя двоичный поиск"""

    # если не стандартизировано, то стандартизировать и нормализовать
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0            # normal_cdf(-10) = (очень близко к) 0
    hi_z,  hi_p  =  10.0, 1            # normal_cdf(10) = (очень близко к) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # взять середину
        mid_p = normal_cdf(mid_z)      # и значение ИФР в этом месте
        if mid_p < p:
            # значение середины все еще слишком низкое, искать выше его
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # значение середины все еще слишком высокое, искать ниже
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(p, n):
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist(p, n, num_points):

    data = [binomial(p, n) for _ in range(num_points)]

    # столбчатая диаграмма, показывающая фактические биномиальные выборки
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # линейный график, показывающий нормальное приближение
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs,ys)
    plt.show()


if __name__ == "__main__":

    #
    # УСЛОВНАЯ ВЕРОЯТНОСТЬ
    #

    both_girls = 0
    older_girl = 0
    either_girl = 0

    random.seed(0)
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == "girl":                        # старшая?
            older_girl += 1
        if older == "girl" and younger == "girl":  # обе?
            both_girls += 1
        if older == "girl" or younger == "girl":   # любая из двух?
            either_girl += 1

    # обе либо старше
    print("P(обе | старшая):",        both_girls / older_girl)   # 0.514 ~ 1/2
    # обе либо ни одной
    print("P(обе | любая из двух): ", both_girls / either_girl)  # 0.342 ~ 1/3


# In[ ]:



