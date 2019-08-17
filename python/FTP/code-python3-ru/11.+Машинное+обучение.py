
# coding: utf-8

# In[1]:



# machine_learning.py

from collections import Counter
import math, random

#
# разбиение данных
#

def split_data(data, prob):
    """разбиение данных на части [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = list(zip(x, y))                        # объединить соответствующие значения
    train, test = split_data(data, 1 - test_pct)  # разбить список пар
    x_train, y_train = list(zip(*train))          # трюк с разъединением списков
    x_test, y_test = list(zip(*test))
    return x_train, x_test, y_train, y_test

#
# точность модели
#

def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)

if __name__ == "__main__":

    print("accuracy(70, 4930, 13930, 981070)", accuracy(70, 4930, 13930, 981070))
    print("precision(70, 4930, 13930, 981070)", precision(70, 4930, 13930, 981070))
    print("recall(70, 4930, 13930, 981070)", recall(70, 4930, 13930, 981070))
    print("f1_score(70, 4930, 13930, 981070)", f1_score(70, 4930, 13930, 981070))


# In[ ]:



