
# coding: utf-8

# In[1]:



# clustering.py

import sys
sys.path.append("../code-python3-ru")

from lib.linear_algebra import squared_distance, vector_mean, distance
import math, random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl

class KMeans:
    """класс выполняет кластеризацию по методу k-средних"""

    def __init__(self, k):
        self.k = k          # число кластеров
        self.means = None   # средние кластеров

    def classify(self, input):
        """вернуть индекс кластера, ближайшего к входящему значению input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):

        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # найти новые назначения
            new_assignments = list(map(self.classify, inputs))

            # если ни одно назначение не изменилось, то завершить
            if assignments == new_assignments:
                return

            # в противном случае сохранить новые назначения
            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # удостовериться, что i_points не пуст, чтобы не делить на 0
                if i_points:
                    self.means[i] = vector_mean(i_points)

def squared_clustering_errors(inputs, k):
    """находит суммарное квадратичное отклонение (ошибку) 
    от k-средних при кластеризации входящих данных"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = list(map(clusterer.classify, inputs))

    return sum(squared_distance(input,means[cluster])
               for input, cluster in zip(inputs, assignments))

def plot_squared_clustering_errors():

    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("Суммарнное квадратичное отклонение")
    plt.show()

#
# применение кластеризации для изменения цвета изображения
#

def recolor_image(input_file, k=5):

    img = mpimg.imread(path_to_png_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels) # обучение может занять некоторое время

    def recolor(pixel):
        cluster = clusterer.classify(pixel) # индекс ближайшего кластера
        return clusterer.means[cluster]     # среднее ближайшего кластера

    new_img = [[recolor(pixel) for pixel in row]
               for row in img]

    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

#
# иерархическая кластеризация
#

def is_leaf(cluster):
    """кластер является листом, если его длина = 1"""
    return len(cluster) == 1

def get_children(cluster):
    """вернуть два дочерних элемента данного кластера,
    если он – объединенный кластер; вызывает исключение,
    если это листовой кластер"""
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]

def get_values(cluster):
    """вернуть значение в кластере (если это листовой кластер)
    или все значения в листовых кластерах под ним (если нет)"""
    if is_leaf(cluster):
        return cluster # это уже одноэлементный кортеж, содержащий значение
    else:
        return [value
                for child in get_children(cluster)
                for value in get_values(child)]

def cluster_distance(cluster1, cluster2, distance_agg=min):
    """найти агрегированное расстояние между элементами 
    кластера cluster1 и элементами кластера cluster2"""
    return distance_agg([distance(input1, input2)
                        for input1 in get_values(cluster1)
                        for input2 in get_values(cluster2)])

def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0] # 1-й элемент 2-элементного кортежа - порядковый номер merge_order 

def bottom_up_cluster(inputs, distance_agg=min):
    # начать с того, что все входы - листовые кластеры (1-элементный кортеж)
    clusters = [(input,) for input in inputs]

    # пока остается более одного кластера...
    while len(clusters) > 1:
        # найти два ближайших кластера
        c1, c2 = min([(cluster1, cluster2)
                     for i, cluster1 in enumerate(clusters)
                     for cluster2 in clusters[:i]],
                     key=lambda p: cluster_distance(p[0], p[1], distance_agg))

        # исключить их из списка кластеров
        clusters = [c for c in clusters if c != c1 and c != c2]

        # объединить их, используя переменную порядкового номера
        # объединения merge_order = число оставшихся кластеров
        merged_cluster = (len(clusters), [c1, c2])

        # и добавить их объединение к списку кластеров
        clusters.append(merged_cluster)

    # когда останется всего один кластер, то вернуть его
    return clusters[0]

def generate_clusters(base_cluster, num_clusters):
    # sначать со списка, состоящего только из базового кластера
    clusters = [base_cluster]

    # продолжать, пока кластеров не достаточно...
    while len(clusters) < num_clusters:
        # выбрать из кластеров тот, который был объединен последним
        next_cluster = min(clusters, key=get_merge_order)
        # исключить его из списка
        clusters = [c for c in clusters if c != next_cluster]
        # и добавить его дочерние элементы к списку
        # (т. е. разъединить его)
        clusters.extend(get_children(next_cluster))

    # когда уже достаточно кластеров...
    return clusters

if __name__ == "__main__":

    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],
              [21,27],[-49,15],[26,13],[-46,5],[-34,-1],
              [11,15],[-49,0],[-22,-16],[19,28],[-12,-8],
              [-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    random.seed(0) # чтобы получить  повторимые результаты
    clusterer = KMeans(3)
    clusterer.train(inputs)
    print("3-средних:")
    print(clusterer.means)
    print()

    random.seed(0)
    clusterer = KMeans(2)
    clusterer.train(inputs)
    print("2-средних:")
    print(clusterer.means)
    print()

    print("квадратичные отклонения как функция от k")

    for k in range(1, len(inputs) + 1):
        print(k, squared_clustering_errors(inputs, k))
    print()


    print("восходящий метод иерархической кластеризации")

    base_cluster = bottom_up_cluster(inputs)
    print(base_cluster)

    print()
    print("3 кластера, min:")
    for cluster in generate_clusters(base_cluster, 3):
        print(get_values(cluster))

    print()
    print("3 кластера, max:")
    base_cluster = bottom_up_cluster(inputs, max)
    for cluster in generate_clusters(base_cluster, 3):
        print(get_values(cluster))


# In[ ]:



