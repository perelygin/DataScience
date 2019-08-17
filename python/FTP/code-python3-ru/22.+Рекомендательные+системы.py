
# coding: utf-8

# In[2]:



# recommender_systems.py

import sys
sys.path.append("../code-python3-ru")

import math, random
from collections import defaultdict, Counter
from lib.linear_algebra import dot

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests).most_common()

def most_popular_new_interests(user_interests, max_results=5):
    suggestions = [(interest, frequency)
                   for interest, frequency in popular_interests
                   if interest not in user_interests]
    return suggestions[:max_results]

#
# коллаборативная фильтрация на основе пользователя
#

def cosine_similarity(v, w):
    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))

unique_interests = sorted(list({ interest
                                 for user_interests in users_interests
                                 for interest in user_interests }))

def make_user_interest_vector(user_interests):
    """при заданном списке интересующих пользователя тем создать вектор,
    чей i-ый элемент равен 1, если unique_interests[i] есть в списке,
    и 0 в противном случае"""
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

user_interest_matrix = list(map(make_user_interest_vector, users_interests))

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                     for interest_vector_i in user_interest_matrix]

def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity)                      # найти других
             for other_user_id, similarity in                 # пользователей с
                enumerate(user_similarities[user_id])         # ненулевым коэфф.
             if user_id != other_user_id and similarity > 0]  # подобия

    return sorted(pairs,                                      # отсортировать их
                  key=lambda pair: pair[1],                   # сперва наиболее
                  reverse=True)                               # похожие


def user_based_suggestions(user_id, include_current_interests=False):
    # суммировать все коэффициенты подобия
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # преобразовать их в сортированный список
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[1],
                         reverse=True)

    # и (может быть) исключить уже имеющиеся темы
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

#
# коллаборативная фильтрация по схожести предметов
#

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_matrix]
                        for j, _ in enumerate(unique_interests)]

interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[1],
                  reverse=True)

def item_based_suggestions(user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]


if __name__ == "__main__":

    print("Популярные темы")
    print(popular_interests)
    print()

    print("Наиболее популярные новые темы")
    print("уже нравится:", ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"])
    print("новые: ", most_popular_new_interests(["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"]))
    print()
    print("уже нравится:", ["R", "Python", "statistics", "regression", "probability"])
    print("новые: ", most_popular_new_interests(["R", "Python", "statistics", "regression", "probability"]))
    print()

    print("Подобие (схожесть) на основе пользователя")
    print("наиболее похожие на пользователя 0")
    print(most_similar_users_to(0))

    print("Рекомендации для пользователя 0")
    print(user_based_suggestions(0))
    print()

    print("Коллаборативная фильтрация по схожести предметов")
    print("наиболее похожие на 'Big Data'")
    print(most_similar_interests_to(0))
    print()

    print("Рекомендации для пользователя 0")
    print(item_based_suggestions(0))


# In[ ]:



