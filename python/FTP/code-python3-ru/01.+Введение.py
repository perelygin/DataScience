
# coding: utf-8

# # Глава 1. Введение

# In[1]:



# introduction.py

import matplotlib as mpl
# на этой стадии книги мы на самом деле еще не установили библиотеку matplotlib,
# закомментируйте эти строки, если хотите
from matplotlib import pyplot as plt
mpl.style.use('ggplot')

##########################
#                        #
# ПОИСК КЛЮЧЕВЫХ ЗВЕНЬЕВ #
#                        #
##########################

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" },
    { "id": 10, "name": "Jen" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]


# сначала назначим каждому пользователю пустой список
# свойство friends содержит друзей для пользователя user
for user in users:
    user["friends"] = []

# и затем заполним списки друзьями
for i, j in friendships:
    # это работает, потому что users[i] - это пользователь, чей id равен i
    users[i]["friends"].append(users[j]) # добавить i как друга для j
    users[j]["friends"].append(users[i]) # добавить j как друга для i

def number_of_friends(user):
    """сколько друзей есть у пользователя user?"""
    return len(user["friends"]) # длина списка id друзей

total_connections = sum(number_of_friends(user)  # общее число связей
                        for user in users)       # 24

num_users = len(users)                           # длина списка пользователей
avg_connections = total_connections / num_users  # среднее число связей = 2.4

######################################
#                                    #
# АНАЛИТИКИ, КОТОРЫХ ВЫ ДОЛЖНЫ ЗНАТЬ #
#                                    #
######################################

# список id друзей пользователя user (плохой вариант функции)
def friends_of_friend_ids_bad(user):
    # "foaf" - это аббревиатура для "friend of a friend", .т.е. друг конкретного друга
    return [foaf["id"]
            for friend in user["friends"]  # для каждого из друзей пользователя
            for foaf in friend["friends"]] # получить всех ЕГО друзей

from collections import Counter # не загружается по умолчанию

# не тот же самый
def not_the_same(user, other_user):
    """два пользователя не одинаковые, если их ключи имеют разные id"""
    return user["id"] != other_user["id"]

# не друзья
def not_friends(user, other_user):
    """other_user - не друг, если он не принадлежит user["friends"], т. е.
    если он not_the_same (не тот же что и все люди в user["friends"])"""
    return all(not_the_same(friend, other_user)
               for friend in user["friends"])

# список id друзей пользователя user
def friends_of_friend_ids(user):
    return Counter(foaf["id"]
                   for friend in user["friends"]  # для каждого моего друга
                   for foaf in friend["friends"]  # подсчитать ИХ друзей,
                   if not_the_same(user, foaf)    # которые не являются мной
                   and not_friends(user, foaf))   # и не мои друзья

print(friends_of_friend_ids(users[3])) # Counter({0: 2, 5: 1})

# интересующие темы
interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# аналитики, которым нравится целевая тема target_interest
def data_scientists_who_like(target_interest):
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]

from collections import defaultdict

# id пользователей по значению темы
# ключи - это интересующие темы,
# значения - это списки из id пользователей, интересующихся этой темой
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# идентификаторы тем по идентификатору пользователя
# ключи - это id пользователей, значения - списки тем для конкретного id
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

# наиболее общие интересующие темы с пользователем user
def most_common_interests_with(user_id):
    return Counter(interested_user_id
        for interest in interests_by_user["user_id"]
        for interested_user_id in users_by_interest[interest]
        if interested_user_id != user_id)

##########################
#                        #
# ЗАРПЛАТЫ И ОПЫТ РАБОТЫ #
#                        #
##########################

# зарплаты и стаж
salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

# Зависимость заработной платы от опыта работы
def make_chart_salaries_by_tenure():
    tenures = [tenure for salary, tenure in salaries_and_tenures]
    salaries = [salary for salary, tenure in salaries_and_tenures]
    plt.scatter(tenures, salaries)
    plt.xlabel("Стаж, лет")
    plt.ylabel("Заработная плата, долл.")
    plt.show()

# зарплата в зависимости от стажа
# ключи - это годы, значения - это списки зарплат для каждого стажа
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# средняя зарплата в зависимости от стажа
# ключи - это годы, каждое значение - это средняя зарплата по этому стажу
average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

# стажная группа
def tenure_bucket(tenure):
    if tenure < 2: return "менее двух"
    elif tenure < 5: return "между двумя и пятью"
    else: return "более пяти"

salary_by_tenure_bucket = defaultdict(list)

# зарплата в зависимости от стажной группы
# ключи = стажные группы, значения = списки зарплат в этой группе
# словарь содержит списки зарплат, соответствующие каждой стажной группе
for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# средняя зарплата по группе
# ключи = стажные группы, значения = средняя зарплата по этой группе
average_salary_by_bucket = {
  tenure_bucket : sum(salaries) / len(salaries)
  for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}


############################
#                          #
# ОПЛАТА ПРЕМИУМ-АККАУНТОВ #
#                          #
############################

# предсказать платежи, исходя из стажа
def predict_paid_or_unpaid(years_experience):
    if years_experience < 3.0: return "оплачено"
    elif years_experience < 8.5: return "не оплачено"
    else: return "оплачено"

###################
#                 #
# ПОПУЛЯРНЫЕ ТЕМЫ #
#                 #
###################

# слова и частотности
words_and_counts = Counter(word
                           for user, interest in interests
                           for word in interest.lower().split())


if __name__ == "__main__":

    print()
    print("######################")
    print("#")
    print("# ПОИСК КЛЮЧЕВЫХ ЗВЕНЬЕВ")
    print("#")
    print("######################")
    print()


    print("всего связей", total_connections)
    print("количество пользователей", num_users)
    print("среднее количество связей", total_connections / num_users)
    print()

    # create a list (user_id, number_of_friends)
    num_friends_by_id = [(user["id"], number_of_friends(user))
                         for user in users]

    print("пользователи, отсортированные по количеству друзей:")
    print(sorted(num_friends_by_id,
                 key=lambda pair: pair[1],                       # по количеству друзей
                 reverse=True))                                  # от большего к меньшему

    print()
    print("######################")
    print("#")
    print("# АНАЛИТИКИ, КОТОРЫХ ВЫ ДОЛЖНЫ ЗНАТЬ")
    print("#")
    print("######################")
    print()


    print("друзья друзей для пользователя 0 (плохой вариант):", friends_of_friend_ids_bad(users[0]))
    print("друзья друзей для пользователя 3:", friends_of_friend_ids(users[3]))

    print()
    print("######################")
    print("#")
    print("# ЗАРПЛАТЫ И ОПЫТ РАБОТЫ")
    print("#")
    print("######################")
    print()

    print("средняя зарплата по стажу", average_salary_by_tenure)
    print("средняя зарплата по стажной группе", average_salary_by_bucket)

    print()
    print("######################")
    print("#")
    print("# НАИБОЛЕЕ ЧАСТО ИСПОЛЬЗУЕМЫЕ СЛОВА")
    print("#")
    print("######################")
    print()

    for word, count in words_and_counts.most_common():
        if count > 1:
            print(word, count)


# In[ ]:



