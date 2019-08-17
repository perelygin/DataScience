
# coding: utf-8

# In[1]:



# decision_trees.py

import matplotlib as mpl

from collections import Counter, defaultdict
from functools import partial
import math, random

def entropy(class_probabilities):
    """при заданном списке вероятностей классов вычислить энтропию"""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    """найти энтропию исходя из этого разбиения данных на подгруппы"""
    total_count = sum(len(subset) for subset in subsets)

    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets )

def group_by(items, key_fn):
    """возвращает defaultdict(list), где каждый элемент item 
    находится в списке, чей ключ равен key_fn(item)"""
    groups = defaultdict(list)
    for item in items:
        key = key_fn(item)
        groups[key].append(item)
    return groups

def partition_by(inputs, attribute):
    """возвращает словарь dict, состоящий из входящих значений inputs, 
    разделенных по атрибуту, 
    каждое входящее значение - это пара (attribute_dict, метка)"""
    return group_by(inputs, lambda x: x[0][attribute])

def partition_entropy_by(inputs,attribute):
    """вычисляет энтропию, соответствующую заданному разбиению"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def classify(tree, input):
    """классифицировать входящие значения, используя заданное дерево ДПР"""

    # если это листовой узел, вернуть его значение
    if tree in [True, False]:
        return tree

    # иначе найти правильное поддерево
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)  # None, если на входе отсутствующий атрибут

    if subtree_key not in subtree_dict: # если для ключа нет поддерева,
        subtree_key = None              # использовать поддерево None

    subtree = subtree_dict[subtree_key] # выбрать соответствующее поддерево
    return classify(subtree, input)     # и использовать для классификации

def build_tree_id3(inputs, split_candidates=None):

    # если это первый проход, то
    # все ключи первоначальных входящих данных — это выделенные претенденты
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # подсчитать число True и False во входящих значениях
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0:                  # если True отсутствуют
        return False                    # вернуть лист False

    if num_falses == 0:                 # если False отсутствуют
        return True                     # вернуть лист True

    if not split_candidates:            # если больше нет кандидатов
        return num_trues >= num_falses  # вернуть лист большинства

    # в противном случае выполнить расщепление по лучшему атрибуту
    best_attribute = min(split_candidates,
        key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]

    # рекурсивно создать поддеревья
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.items() }

    subtrees[None] = num_trues > num_falses # случай по умолчанию

    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


if __name__ == "__main__":

    inputs = [
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
        ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
        ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
        ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
        ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
        ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
    ]

    for key in ['level','lang','tweets','phd']:
        print(key, partition_entropy_by(inputs, key))
    print()

    senior_inputs = [(input, label)
                     for input, label in inputs if input["level"] == "Senior"]

    for key in ['lang', 'tweets', 'phd']:
        print(key, partition_entropy_by(senior_inputs, key))
    print()

    print("создаем дерево")
    tree = build_tree_id3(inputs)
    print(tree)

    print("Junior / Java / tweets / no phd", classify(tree,
        { "level" : "Junior",
          "lang" : "Java",
          "tweets" : "yes",
          "phd" : "no"} ))

    print("Junior / Java / tweets / phd", classify(tree,
        { "level" : "Junior",
                 "lang" : "Java",
                 "tweets" : "yes",
                 "phd" : "yes"} ))

    print("Intern", classify(tree, { "level" : "Intern" } ))
    print("Senior", classify(tree, { "level" : "Senior" } ))


# In[ ]:



