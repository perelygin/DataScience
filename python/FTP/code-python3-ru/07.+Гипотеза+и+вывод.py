
# coding: utf-8

# In[1]:



# hypothesis_and_inference.py

import sys
sys.path.append("../code-python3-ru")

from lib.probability import normal_cdf, inverse_normal_cdf
import math, random

def normal_approximation_to_binomial(n, p):
    """находит mu и sigma, которые соответствуют binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

#####
#
# вероятности, что нормальная случайная величина лежит в интервале
#
######

# вероятность, что реализованное значение нормальной величины лежит ниже
normal_probability_below = normal_cdf

# вероятность, что оно лежит выше, если оно не ниже его
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

# вероятность, что оно лежит между, если оно меньше hi, но не ниже lo
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# вероятность, что оно лежит за пределами, если оно не внутри
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

######
#
#  границы нормальной величины
#
######

def normal_upper_bound(probability, mu=0, sigma=1):
    """возвращает z, для которого P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1):
    """возвращает z, для которого P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """возвращает симметричные (вокруг среднего значения) границы,
    в пределах которых содержится указанная вероятность"""
    tail_probability = (1 - probability) / 2

    # верхняя граница должна иметь значение хвостовой вероятности
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # нижняя граница должна иметь значение хвостовой вероятности
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # если x больше среднего значения, то значения в хвосте больше x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # если x меньше среднего значения, то значения в хвосте меньше x
        return 2 * normal_probability_below(x, mu, sigma)

def count_extreme_values():
    extreme_value_count = 0
    for _ in range(100000):
        num_heads = sum(1 if random.random() < 0.5 else 0    # подсчитать число орлов
                        for _ in range(1000))                # при 1000 бросках
        if num_heads >= 530 or num_heads <= 470:             # подсчитать, как часто
            extreme_value_count += 1                         # число - 'предельно'

    return extreme_value_count / 100000

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

##
#
# Подгонка p-значения
#
##

def run_experiment():
    """бросить уравновешенную монету 1000 раз, True = орлы, False = решки"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment):
    """используя 5%-ые уровни значимости"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

##
#
# проведение A/B-тестирования
#
##

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

##
#
# Байесовский статистический вывод
#
##

def B(alpha, beta):
    """нормализующая константа, благодаря которой сумма вероятностей равна 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:          # за пределами [0, 1] нет веса
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


if __name__ == "__main__":

    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    print("mu_0", mu_0)
    print("sigma_0", sigma_0)
    print("normal_two_sided_bounds(0.95, mu_0, sigma_0)", normal_two_sided_bounds(0.95, mu_0, sigma_0))
    print()
    print("Мощность критерия")

    print("95% границы на основе допущения, что p = 0.5")

    lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print("lo", lo)
    print("hi", hi)

    print("фактические mu и sigma на основе p = 0.55")
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
    print("mu_1", mu_1)
    print("sigma_1", sigma_1)

    # ошибка 2-ого рода означает, что нам не удастся отклонить нулевую гипотезу,
    # которая случится, когда X все еще лежит в нашем исходном интервале
    type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
    power = 1 - type_2_probability # 0.887

    print("вероятность ошибки 2-ого рода", type_2_probability)
    print("мощность", power)
    print

    print("Односторонний критерий")
    hi = normal_upper_bound(0.95, mu_0, sigma_0)
    print("hi", hi) # равно 526 (< 531, т.к. нам нужно больше вероятности в верхнем хвосте)
    type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
    power = 1 - type_2_probability # = 0.936
    print("вероятность ошибки 2-ого рода", type_2_probability)
    print("мощность", power)
    print()

    print("two_sided_p_value(529.5, mu_0, sigma_0)", two_sided_p_value(529.5, mu_0, sigma_0))

    print("two_sided_p_value(531.5, mu_0, sigma_0)", two_sided_p_value(531.5, mu_0, sigma_0))

    print("upper_p_value(525, mu_0, sigma_0)", upper_p_value(525, mu_0, sigma_0))
    print("upper_p_value(527, mu_0, sigma_0)", upper_p_value(527, mu_0, sigma_0))
    print()

    print("Подгонка p-значения")

    random.seed(0)
    experiments = [run_experiment() for _ in range(1000)]
    num_rejections = len([experiment
                          for experiment in experiments
                          if reject_fairness(experiment)])

    print(num_rejections, "отклонений из 1000")
    print()

    print("A/B-тестирование")
    z = a_b_test_statistic(1000, 200, 1000, 180)
    print("a_b_test_statistic(1000, 200, 1000, 180)", z)
    print("p-значение", two_sided_p_value(z))
    z = a_b_test_statistic(1000, 200, 1000, 150)
    print("a_b_test_statistic(1000, 200, 1000, 150)", z)
    print("p-значение", two_sided_p_value(z))


# In[ ]:



