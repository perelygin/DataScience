
# coding: utf-8

# In[1]:



# neural_networks.py

import sys
sys.path.append("../code-python3-ru")

from collections import Counter
from functools import partial
from lib.linear_algebra import dot
import math, random
import matplotlib
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """возвращает 1, если перцептрон 'активизируется', и 0, если нет"""
    return step_function(dot(weights, x) + bias)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    """принимает нейронную сеть (как список списков списков весов) и
    вектор входящих сигналов;
    возвращает результат прямого распространения входящих сигналов"""

    outputs = []

    for layer in neural_network:

        input_with_bias = input_vector + [1]             # добавить величину смещения
        output = [neuron_output(neuron, input_with_bias) # вычислить результат
                  for neuron in layer]                   # для этого нейрона
        outputs.append(output)                           # и запомнить его

        # входом для следующего слоя становится
        # вектор результатов текущего слоя
        input_vector = output

    return outputs

def backpropagate(network, input_vector, target):

    hidden_outputs, outputs = feed_forward(network, input_vector)

    # из производной сигмоидальной функции взято output * (1 - output)
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # понейронно скорректировать веса для слоя выходов (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # распространить ошибки на скрытый слой, двигаясь в обратную сторону
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # понейронно скорректировать веса для скрытого слоя (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def patch(x, y, hatch, color):
    """вернуть объект 'patch' библиотеки matplotlib с указанными
    координатами, шаблоном штриховки и цветом"""
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)


def show_weights(neuron_idx):
    weights = network[0][neuron_idx]
    abs_weights = list(map(abs, weights)) #UPD в оригинале map(abs, weights)

    grid = [abs_weights[row:(row+5)]     # преобразовать веса в матрицу 5 x 5 
            for row in range(0,25,5)]    # [weights[0:5], ..., weights[20:25]]

    ax = plt.gca()                       # для штриховки нужен объект axis (ось)

    ax.imshow(grid,                      # здесь так же как в plt.imshow
              cmap=matplotlib.cm.binary, # использовать шкалу ч/б цвета
              interpolation='none')      # рисовать блоками

    # заштриховать отрицательные веса
    for i in range(5):               # строка
        for j in range(5):           # столбец
            if weights[5*i + j] < 0: # строка i, столбец j = weights[5*i + j]
                # добавить ч/б штриховку, чтобы было видно на темном или светлом
                ax.add_patch(patch(j, i, '/', "white"))
                ax.add_patch(patch(j, i, '\\', "black"))
    plt.show()

if __name__ == "__main__":

    raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",

          """11111
             ....1
             11111
             1....
             11111""",

          """11111
             ....1
             11111
             ....1
             11111""",

          """1...1
             1...1
             11111
             ....1
             ....1""",

          """11111
             1....
             11111
             ....1
             11111""",

          """11111
             1....
             11111
             1...1
             11111""",

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             1...1
             11111
             1...1
             11111""",

          """11111
             1...1
             11111
             ....1
             11111"""]

    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]

    inputs = list(map(make_digit, raw_digits))  

    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(0)   # чтобы получить повторимые результаты
    input_size = 25  # каждое входящее значение - это вектор длиной 25 элементов
    num_hidden = 5   # 5 нейронов в скрытом слое
    output_size = 10 # 10 значений на выходе для каждого входящего

    # каждый скрытый нейрон имеет один вес для входного нейрона
    # плюс вес смещения
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # каждый выходной нейрон имеет один вес для скрытого нейрона
    # плюс вес смещения
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    # сеть начинает работу со случайными весами
    network = [hidden_layer, output_layer]

    # для схождения сети 10000 итераций, по-видимому, будет достаточно
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    def predict(input):
        return feed_forward(network, input)[-1]

    for i, input in enumerate(inputs):
        outputs = predict(input)
        print(i, [round(p,2) for p in outputs])

    print(""".@@@.
...@@
..@@.
...@@
.@@@.""")
    print([round(x, 2) for x in
          predict(  [0,1,1,1,0,    # .@@@.
                     0,0,0,1,1,    # ...@@
                     0,0,1,1,0,    # ..@@.
                     0,0,0,1,1,    # ...@@
                     0,1,1,1,0])]) # .@@@.
    print()

    print(""".@@@.
@..@@
.@@@.
@..@@
.@@@.""")
    print([round(x, 2) for x in
          predict(  [0,1,1,1,0,    # .@@@.
                     1,0,0,1,1,    # @..@@
                     0,1,1,1,0,    # .@@@.
                     1,0,0,1,1,    # @..@@
                     0,1,1,1,0])]) # .@@@.
    print()

    show_weights(0)

