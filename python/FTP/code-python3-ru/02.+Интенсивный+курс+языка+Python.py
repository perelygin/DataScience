
# coding: utf-8

# # Интенсивный курс языка Python

# ## Основы языка Python
# #### на примере Python 3

# ### Пробельные символы

# In[8]:

# пример отступов во вложенных циклах for
for i in [1, 2, 3, 4, 5]:
    print(i)                   # первая строка в блоке for i
    for j in [1, 2, 3, 4, 5]:
        print(j, end=" ")        # первая строка в блоке for j
        print(i + j, end=" ")    # последняя строка в блоке for j
    print(i)                   # последняя строка в блоке for i
print("циклы закончились")


# In[9]:

# пример многословного выражения
long_winded_computation = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 +
                          11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)


# In[10]:

# список списков
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# такой список списков легче читается
easier_to_read_list_of_lists = [ [1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9] ]


# In[11]:

two_plus_three = 2 +                  3


# In[20]:

for i in [1, 2, 3, 4, 5]:

    # обратите внимание на пустую строку (в консоли интерпретатора поднимает исключение!)
    print(i)


# ### Модули

# In[ ]:

import re
my_regex = re.compile("[0-9]+", re.I)


# In[ ]:

import re as regex
my_regex = regex.compile("[0-9]+", regex.I)


# In[ ]:

import matplotlib.pyplot as plt


# In[11]:

from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()


# In[ ]:

match = 10
from re import *   # в модуле re есть функция match
print match        # "<function re.match>"


# ### Арифметические операции

# In[ ]:

# в Python 2 частное округляется вниз, 
# поэтому в случае вещественного результата деления 
# следует импортировать division
from __future__ import division


# ### Функции

# In[13]:

def double(x):
    """здесь, когда нужно, размещают
    многострочный документирующий комментарий docstring,
    который поясняет, что именно функция вычисляет.
    Например, данная функция умножает входящее значение на 2"""
    return x * 2


# In[14]:

# применить функцию f к единице
def apply_to_one(f):
    """вызывает функцию f с единицей в качестве аргумента"""
    return f(1)

my_double = double           # ссылка на ранее определенную функцию
x = apply_to_one(my_double)  # = 2

print(x)


# In[15]:

y = apply_to_one(lambda x: x + 4)     # = 5

print(y)


# In[ ]:

another_double = lambda x: 2 * x      # так не делать
def another_double(x): return 2 * x   # лучше так


# In[19]:

def my_print(message="мое сообщение по умолчанию"):
    print(message)

my_print("привет")  # напечатает 'привет'
my_print()          # напечатает 'мое сообщение по умолчанию'


# In[17]:

# функция вычитания
def subtract(a=0, b=0):
    return a - b

subtract(10, 5)    # возвращает 5
subtract(0, 5)     # возвращает -5
subtract(b=5)      # то же, что и в предыдущем примере


# ### Строки

# In[ ]:

single_quoted_string = 'наука о данных'   # одинарные
double_quoted_string = "наука о данных"   # двойные


# In[16]:

tab_string = "\t"        # обозначает символ табуляции
len(tab_string)          # = 1


# In[15]:

not_tab_string = r"\t"   # обозначает символы '\' и 't'
len(not_tab_string)      # = 2


# In[14]:

multi_line_string = """Это первая строка.
это вторая строка
а это третья строка"""


# ### Исключения

# In[24]:

try:
    print(0 / 0)
except ZeroDivisionError:
    print("нельзя делить на ноль")


# ### Списки

# In[17]:

integer_list = [1, 2, 3]                       # список целых чисел
heterogeneous_list = ["строка", 0.1, True]     # разнородный список
list_of_lists = [ integer_list, heterogeneous_list, [] ]  # список списков

list_length = len(integer_list)  # длина списка = 3
list_sum    = sum(integer_list)  # сумма значений в списке = 6

print(list_sum)


# In[18]:

x = list(range(10)) # задает список [0, 1, ..., 9]
zero = x[0]   # = 0, списки нуль-индексные, т. е. индекс 1-го элемента = 0
one = x[1]    # = 1
nine = x[-1]  # = 9, по-питоновски взять последний элемент
eight = x[-2] # = 8, по-питоновски взять предпоследний элемент
x[0] = -1     # теперь x = [-1, 1, 2, 3, ..., 9]

print(x)


# In[27]:

first_three  = x[:3]   # первые три = [-1, 1, 2]
three_to_end = x[3:]   # с третьего до конца = [3, 4, ..., 9]
one_to_four  = x[1:5]  # с первого по четвертый = [1, 2, 3, 4]
last_three   = x[-3:]  # последние три = [7, 8, 9]
without_first_and_last = x[1:-1]  # без первого и последнего = [1, 2, ..., 8]
copy_of_x = x[:]       # копия списка x = [-1, 1, 2, ..., 9]


# In[28]:

1 in [1, 2, 3]    # True
0 in [1, 2, 3]    # False


# In[29]:

x = [1, 2, 3]
x.extend([4, 5, 6])  # теперь x = [1, 2, 3, 4, 5, 6]


# In[30]:

x = [1, 2, 3]
y = x + [4, 5, 6]    # y = [1, 2, 3, 4, 5, 6]; x не изменился


# In[31]:

x = [1, 2, 3]
x.append(0)          # теперь x = [1, 2, 3, 0]
y = x[-1]            # = 0
z = len(x)           # = 4


# In[32]:

x, y = [1, 2]        # теперь x = 1, y = 2


# In[33]:

_, y = [1, 2]        # теперь y == 2, первый элемент не нужен


# ### Кортежи

# In[38]:

my_list = [1, 2]    # задать список
my_tuple = (1, 2)   # задать кортеж
other_tuple = 3, 4  # еще один кортеж
my_list[1] = 3      # теперь my_list = [1, 3]

try:
    my_tuple[1] = 3
except TypeError:
    print("кортеж изменять нельзя")


# In[39]:

# функция возвращает сумму и произведение двух параметров
def sum_and_product(x, y):
    return (x + y),(x * y)

sp = sum_and_product(2, 3)     # = (5, 6)
s, p = sum_and_product(5, 10)  # s = 15, p = 50


# In[40]:

x, y = 1, 2   # теперь x = 1, y = 2
x, y = y, x   # обмен переменными по-питоновски; теперь x = 2, y = 1


# ### Словари

# In[41]:

empty_dict = {}              # задать словарь по-питоновски
empty_dict2 = dict()         # не совсем по-питоновски
grades = {"Joel" : 80, "Tim" : 95}  # литерал словаря (оценки за экзамены)


# In[42]:

joels_grade = grades["Joel"]        # = 80


# In[44]:

try:
    kates_grade = grades["Kate"]
except KeyError:
    print("оценки для Кэйт отсутствуют!")


# In[45]:

joel_has_grade = "Joel" in grades      # True
kate_has_grade = "Kate" in grades      # False


# In[46]:

joels_grade = grades.get("Joel", 0)    # = 80
kates_grade = grades.get("Kate", 0)    # = 0
no_ones_grade = grades.get("No One")   # значение по умолчанию = None


# In[47]:

grades["Tim"] = 99             # заменяет старое значение
grades["Kate"] = 100           # добавляет третью запись
num_students = len(grades)     # = 3


# In[48]:

tweet = {
    "user" : "joelgrus",
    "text" : "Наука о данных - потрясающая тема",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}


# In[49]:

tweet_keys   = tweet.keys()    # список ключей
tweet_values = tweet.values()  # список значений
tweet_items  = tweet.items()   # список кортежей (ключ, значение)

"user" in tweet_keys        # True, но использует медленное in списка
"user" in tweet             # по-питоновски, использует быстрое in словаря
"joelgrus" in tweet_values  # True


# ### Словарь defaultdict

# In[51]:

# частотности слов
word_counts = {}   
document = {}  # некий документ, здесь он пустой
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1


# In[52]:

word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1


# In[53]:

word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1


# In[54]:

from collections import defaultdict

word_counts = defaultdict(int)     # int() возвращает 0
for word in document:
    word_counts[word] += 1


# In[61]:

dd_list = defaultdict(list)  # list() возвращает пустой список
dd_list[2].append(1)         # теперь dd_list содержит {2: [1]}
print(dd_list[2])

dd_dict = defaultdict(dict)  # dict() возвращает пустой словарь dict
dd_dict["Joel"]["City"] = "Seattle"  # { "Joel" : { "City" : Seattle"}}
print(dd_dict["Joel"]["City"])

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1            # теперь dd_pair содержит {2: [0,1]}
print(dd_pair[2][1])


# ### Словарь Counter

# In[62]:

from collections import Counter
c = Counter([0, 1, 2, 0])    # в результате c = { 0 : 2, 1 : 1, 2 : 1 }
print(c)


# In[57]:

# лучший вариант подсчета частотностей слов
word_counts = Counter(document)


# In[58]:

# напечатать 10 наиболее встречаемых слов и их частотность (встречаемость)
for word, count in word_counts.most_common(10):
    print(word, count)


# ### Множества

# In[63]:

s = set()     # задать пустое множество

s.add(1)      # теперь s = { 1 }
s.add(2)      # теперь s = { 1, 2 }
s.add(2)      # s как и прежде = { 1, 2 }

x = len(s)    # = 2
y = 2 in s    # = True
z = 3 in s    # = False

print(x,y,z)


# In[65]:

# некий список слов
hundreds_of_other_words = []  # здесь это множество пустое
# список стоп-слов
stopwords_list = ["a","an","at"] + hundreds_of_other_words + ["yet", "you"]
"zip" in stopwords_list    # False, но проверяется каждый элемент

# множество стоп-слов
stopwords_set = set(stopwords_list)
"zip" in stopwords_set     # очень быстрая проверка


# In[66]:

item_list = [1, 2, 3, 1, 2, 3]         # список
num_items = len(item_list)             # количество = 6
item_set = set(item_list)              # вернет множество {1, 2, 3}
num_distinct_items = len(item_set)     # число недублирующихся = 3
distinct_item_list = list(item_set)    # назад в список = [1, 2, 3]


# ### Управляющие конструкции

# In[ ]:

if 1 > 2:
    message = "если 1 была бы больше 2..."
elif 1 > 3:
    message = "elif означает 'else if'"
else:
    message = "когда все предыдущие условия не выполняются, используется else"


# In[ ]:

parity = "четное" if x % 2 == 0 else "нечетное"


# In[67]:

x = 0
while x < 10:
    print(x, "меньше 10")
    x += 1


# In[68]:

for x in range(10):
    print(x, "меньше 10")


# In[69]:

for x in range(10):
    if x == 3:
        continue  # перейти сразу к следующей итерации
    if x == 5:
        break     # выйти из цикла
    print(x)


# ### Истинность

# In[70]:

one_is_less_than_two = 1 < 2         # = True
true_equals_false = True == False    # = False


# In[71]:

x = None
print(x == None)    # напечатает True, но это не по-питоновски
print(x is None)    # напечатает True по-питоновски


# In[73]:

def some_function_that_returns_a_string(): return ""

s = some_function_that_returns_a_string() # возвращает строковое значение
if s:
    first_char = s[0]   # первый символ в строке
else:
    first_char = ""


# In[ ]:

first_char = s and s[0]


# In[ ]:

safe_x = x or 0     # безопасный способ


# In[74]:

all([True, 1, { 3 }])    # True
all([True, 1, {}])       # False, {} = ложное
any([True, 1, {}])       # True, True = истинное
all([])      # True, ложные элементы в списке отсутствуют
any([])      # False, истинные элементы в списке отсутствуют


# ## Не совсем основы

# ### Сортировка

# In[75]:

x = [4,1,2,3]
y = sorted(x)    # после сортировки = [1, 2, 3, 4], x не изменился
x.sort()         # теперь x = [1, 2, 3, 4]

print(x)


# In[80]:

# сортировать список по абсолютному значению в убывающем порядке
x = sorted([-4,1,-2,3], key=abs, reverse=True)    # = [-4,3,-2,1]

print(x)

# сортировать слова и их частотности по убывающему значению частот
wc = sorted(word_counts.items(), key=lambda word, count: count, reverse=True)

print(wc)


# ### Генераторы последовательностей

# In[81]:

# четные числа
even_numbers = [x for x in range(5) if x % 2 == 0]  # [0, 2, 4]
# квадраты чисел
squares    = [x * x for x in range(5)]              # [0, 1, 4, 9, 16]
# квадраты четных чисел
even_squares = [x * x for x in even_numbers]        # [0, 4, 16]


# In[82]:

# словарь с квадратами чисел
square_dict = { x : x * x for x in range(5) } # { 0:0, 1:1, 2:4, 3:9, 4:16 }
# множество с квадратами чисел
square_set  = { x * x for x in [1, -1] }    # { 1 }


# In[83]:

# нули
zeroes = [0 for _ in even_numbers] # имеет ту же длину, что и even_numbers


# In[84]:

# пары
pairs = [(x, y)
         for x in range(10)
         for y in range(10)]    # 100 пар (0,0) (0,1)... (9,8), (9,9)


# In[85]:

# пары с возрастающим значением
increasing_pairs = [(x, y)    # только пары с x < y,
                    for x in range(10)            # range(мин, макс) равен
                    for y in range(x + 1, 10)]    # [мин, мин + 1, ..., макс - 1]


# ### Функции-генераторы и генераторные выражения

# In[87]:

def lazy_range(n):
    """ленивая версия диапазона range"""
    i = 0
    while i < n:
        yield i 
        i += 1


# In[88]:

def do_something_with(i): print(i, end=" ")
    
for i in lazy_range(10): do_something_with(i)


# In[89]:

# натуральные числа
def natural_numbers():
    """возвращает 1, 2, 3, ..."""
    n = 1
    while True:
        yield n 
        n += 1


# In[90]:

# ленивый список четных чисел меньше 20
lazy_evens_below_20 = (i for i in lazy_range(20) if i % 2 == 0)


# ### Случайные числа

# In[92]:

import random

# четыре равномерные случайные величины
four_uniform_randoms = [random.random() for _ in range(4)]

# [0.8444218515250481,     # random.random() производит числа
#  0.7579544029403025,     # равномерно в интервале между 0 и 1
#  0.420571580830845,      # функция random будет применяться
#  0.25891675029296335]    # наиболее часто

print(four_uniform_randoms)


# In[94]:

random.seed(10)    # задать случайную последовательность, установив в 10
print(random.random())      # 0.57140259469
random.seed(10)             # переустановить seed в 10
print(random.random())      # опять 0.57140259469


# In[98]:

random.randrange(10)        # произвольно выбрать из range(10) = [0, 1, ..., 9]
random.randrange(3, 6)      # произвольно выбрать из range(3, 6) = [3, 4, 5]

print(random.randrange(10))
print(random.randrange(3, 6))


# In[106]:

up_to_ten = list(range(10))  # задать последовательность из 10 элементов
print(up_to_ten)

random.shuffle(up_to_ten)
print(up_to_ten)
# [2, 5, 1, 9, 7, 3, 8, 6, 4, 0] (фактические результаты могут отличаться)


# In[108]:

# мой лучший друг
my_best_friend = random.choice(["Алиса", "Боб", "Чарли"])  # сейчас "Боб"

print(my_best_friend)


# In[110]:

# лотерейные номера
lottery_numbers = list(range(60))

# выигрышные номера (пример выборки без возврата)
winning_numbers = random.sample(lottery_numbers, 6)  # [16, 36, 10, 6, 25, 9]

print(winning_numbers)


# In[111]:

# список из четерех элементов (пример выборки с возвратом)
four_with_replacement = [random.choice(list(range(10)))
                         for _ in range(4)]

print(four_with_replacement) # [9, 4, 4, 2]


# ### Регулярные выражения

# In[112]:

import re

print(all([                              # все они - истинные, т. к.
  not re.match("a", "cat"),              # слово 'cat' не начинается с 'a'
  re.search("a", "cat"),                 # в слове 'cat' есть 'a'
  not re.search("c", "dog"),             # в слове 'dog' нет 'c'
  3 == len(re.split("[ab]", "carbs")),   # разбивка по a или b
                                         # в ['c','r','s']
  "R-D-" == re.sub("[0-9]", "-", "R2D2") # замена цифр дефисами
]))  # напечатает True


# ### Объектно-ориентированное программирование

# In[113]:

# по традиции классам назначают имена с заглавной буквы
class Set:

    # ниже идут компонентные функции
    # каждая берет первый параметр "self" (еще одно правило),
    # который ссылается на конкретный используемый объект класса Set
    def __init__(self, values=None):
        """Это конструктор.
        Вызывается при создании нового объекта класса Set и
        используется следующим образом:
        s1 = Set()           # пустое множество
        s2 = Set([1,2,2,3])  # инициализировать значениями"""

        self.dict = {} # каждый экземпляр имеет собственное свойство dict,
                       # которое используется для проверки
                       # на принадлежность элементов множеству

        if values is not None:
            for value in values: self.add(value)

    def __repr__(self):
        """это строковое представление объекта Set,
        которое выводится в оболочке или передается в функцию str()"""
        return "Set: " + str(self.dict.keys())

    # принадлежность представлена ключом в словаре self.dict
    # со значением True
    def add(self, value):
        self.dict[value] = True

    # значение принадлежит множеству, если его ключ имеется в словаре
    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]


# In[115]:

s = Set([1,2,3])

s.add(4)
print(s.contains(4))    # True

s.remove(3)
print(s.contains(3))    # False


# ### Инструменты функционального программирования

# In[116]:

# возведение в степень power
def exp(base, power):
    return base ** power


# In[117]:

# двойка в степени power
def two_to_the(power):
    return exp(2, power)


# In[118]:

from functools import partial
# возвращает функцию с частичным приложением аргументов
two_to_the = partial(exp, 2)     # теперь это функция одной переменной
print(two_to_the(3))             # 8


# In[119]:

square_of = partial(exp, power=2)   # квадрат числа
print(square_of(3))                 # 9


# In[127]:

def double(x):
    return 2 * x

xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs]     # [2, 4, 6, 8]
print(twice_xs)

twice_xs = list(map(double, xs))       # то же, что и выше
print(twice_xs)

list_doubler = partial(map, double)    # удвоитель списка
twice_xs = list(list_doubler(xs))      # снова [2, 4, 6, 8]
print(twice_xs)


# In[129]:

# перемножить аргументы
def multiply(x, y): return x * y

products = list(map(multiply, [1, 2], [4, 5]))  # [1 * 4, 2 * 5] = [4, 10]
print(products)


# In[131]:

# проверка четности
def is_even(x):
    """True, если x – четное; False, если x - нечетное"""
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)]    # список четных чисел = [2, 4]
print(x_evens)

x_evens = list(filter(is_even, xs))        # то же, что и выше
print(x_evens)

list_evener = partial(filter, is_even)     # функция, которая фильтрует список
x_evens = list(list_evener(xs))            # снова [2, 4]
print(x_evens)


# In[137]:

from functools import reduce

xs = [1, 2, 3, 4]
x_product = reduce(multiply, xs)          # = 1 * 2 * 3 * 4 = 24
print(x_product)

list_product = partial(reduce, multiply)   # функция, которая упрощает список
print(x_product)

x_product = list_product(xs)               # снова = 24
print(x_product)


# ### Функция enumerate

# In[140]:

documents = []  # список неких документов; здесь он пустой

# не по-питоновски
for i in range(len(documents)):
    document = documents[i]
    do_something(i, document)

def do_something(i, doc): print(i, doc)
    
# тоже не по-питоновски
i = 0
for document in documents:
    do_something(i, document)
    i += 1


# ### Функция zip и распаковка аргументов

# In[3]:

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
pairs = list(zip(list1, list2))    # = [('a', 1), ('b', 2), ('c', 3)]

print(pairs)


# In[4]:

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

print(letters, numbers)


# In[5]:

xs = list(zip(('a', 1), ('b', 2), ('c', 3)))

print(xs)


# In[6]:

def add(a, b): return a + b

add(1, 2)      # вернет 3
add([1, 2])    # ошибка TypeError!
add(*[1, 2])   # вернет 3


# ### Переменные args и kwargs

# In[7]:

# удвоитель
def doubler(f):
    def g(x):
        return 2 * f(x)
    return g


# In[8]:

def f1(x):
    return x + 1

g = doubler(f1)
print(g(3))       # 8 или ( 3 + 1) * 2
print(g(-1))      # 0 или (-1 + 1) * 2


# In[9]:

def f2(x, y):
    return x + y

g = doubler(f2)
print(g(1, 2))    # TypeError: g() принимает ровно 1 аргумент (задано 2)


# In[10]:

def magic(*args, **kwargs):
    print("безымянные аргументы:", args)
    print("аргументы по ключу:", kwargs)

magic(1, 2, key="word", key2="word2")

# напечатает
# безымянные аргументы: (1, 2)
# аргументы по ключу: {'key2': 'word2', 'key': 'word'}


# In[ ]:



