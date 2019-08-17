
# coding: utf-8

# # Глава 24. Алгоритм MapReduce

# In[2]:



# mapreduce.py

import sys
sys.path.append("../code-python3-ru")

import math, random, re, datetime
from collections import defaultdict, Counter
from functools import partial
from lib.naive_bayes import tokenize

def word_count_old(documents):
    """подсчет частотности слов без использования технологии MapReduce"""
    return Counter(word
        for document in documents
        for word in tokenize(document))

def wc_mapper(document):
    """для каждого слова в документе, генерировать (слово,1)"""
    for word in tokenize(document):
        yield (word, 1)

def wc_reducer(word, counts):
    """суммировать количества появлений слова"""
    yield (word, sum(counts))

def word_count(documents):
    """подсчитать слова во входящих документах при помощи MapReduce"""

    # место для хранения сгруппированных значений
    collector = defaultdict(list)

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.items()
            for output in wc_reducer(word, counts)]

def map_reduce(inputs, mapper, reducer):
    """применяет MapReduce ко входящим данным, используя
       проектор mapper и редуктор reducer"""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output
            for key, values in collector.items()
            for output in reducer(key,values)]

def reduce_with(aggregation_fn, key, values):
    """свертывает пару 'ключ-значение', применяя
    агрегирующую функцию aggregation_fn к значениям"""
    yield (key, aggregation_fn(values))

def values_reducer(aggregation_fn):
    """превращает функцию (значения -> выход) в функцию свертки reducer,
    которая преобразует (ключ, значениея) -> (ключ, выход)"""
    return partial(reduce_with, aggregation_fn)

sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values: len(set(values)))

#
# Анализ обновлений ленты новостей 
#

status_updates = [
    {"id": 1,
     "username" : "joelgrus",
     "text" : "Is anyone interested in a data science book?",
     "created_at" : datetime.datetime(2013, 12, 21, 11, 47, 0),
     "liked_by" : ["data_guy", "data_gal", "bill"] },
    # добавьте свои собственные
]

def data_science_day_mapper(status_update):
    """для дня недели возвращает (day_of_week, 1), если
    обновление ленты status_update содержит "data science" """
    if "data science" in status_update["text"].lower():
        day_of_week = status_update["created_at"].weekday()
        yield (day_of_week, 1)

data_science_days = map_reduce(status_updates,
                               data_science_day_mapper,
                               sum_reducer)

def words_per_user_mapper(status_update):
    user = status_update["username"]
    for word in tokenize(status_update["text"]):
        yield (user, (word, 1))

def most_popular_word_reducer(user, words_and_counts):
    """при заданной последовательности пар (слово, частота),
    вернуть слово с наивысшей итоговой частотой"""

    word_counts = Counter()
    for word, count in words_and_counts:
        word_counts[word] += count

    word, count = word_counts.most_common(1)[0]

    yield (user, (word, count))

user_words = map_reduce(status_updates,
                        words_per_user_mapper,
                        most_popular_word_reducer)

def liker_mapper(status_update):
    user = status_update["username"]
    for liker in status_update["liked_by"]:
        yield (user, liker)

distinct_likers_per_user = map_reduce(status_updates,
                                      liker_mapper,
                                      count_distinct_reducer)


#
# умножение матриц
#

def matrix_multiply_mapper(m, element):
    """m - это общая размерность (столбцы A, строки B)
    элемент - это кортеж (имя матрицы matrix_name, i, j, значение value)"""
    matrix, i, j, value = element

    if matrix == "A":
        for column in range(m):
            # A_ij - это j-ая запись в сумме для каждого C_i_column
            yield((i, column), (j, value))
    else:
        for row in range(m):
            # B_ij — это i-ая запись в сумме для каждого C_row_j
            yield((row, j), (i, value))

def matrix_multiply_reducer(m, key, indexed_values):
    results_by_index = defaultdict(list)
    for index, value in indexed_values:
        results_by_index[index].append(value)

    # суммировать все произведения позиций с двумя результатами
    sum_product = sum(results[0] * results[1]
                      for results in results_by_index.values()
                      if len(results) == 2)

    if sum_product != 0.0:
        yield (key, sum_product)

if __name__ == "__main__":

    documents = ["data science", "big data", "science fiction"]

    wc_mapper_results = [result
                         for document in documents
                         for result in wc_mapper(document)]

    print("результаты wc_mapper")
    print(wc_mapper_results)
    print()

    print("результаты подсчета частотностей слов")
    print(word_count(documents))
    print()

    print("подсчет частотностей слов при помощи функции map_reduce")
    print(map_reduce(documents, wc_mapper, wc_reducer))
    print()

    print("дни науки о данных")
    print(data_science_days)
    print()

    print("слова пользователя")
    print(user_words)
    print()

    print("отдельные предпочтения по пользователю")
    print(distinct_likers_per_user)
    print()

    # matrix multiplication

    entries = [("A", 0, 0, 3), ("A", 0, 1,  2),
               ("B", 0, 0, 4), ("B", 0, 1, -1), ("B", 1, 0, 10)]
    mapper = partial(matrix_multiply_mapper, 3)
    reducer = partial(matrix_multiply_reducer, 3)

    print("умножение матриц на основе алгоритма map-reduce")
    print("записи:", entries)
    print("результат:", map_reduce(entries, mapper, reducer))


# In[ ]:



