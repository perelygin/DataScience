
# coding: utf-8

# In[1]:



# egrep.py

import sys, re

if __name__ == "__main__":

    # sys.argv - это список аргументов командной строки
    # sys.argv[0] - это название самой программы
    # sys.argv[1] будет регулярным выражением, заданным в командной строке
    regex = sys.argv[1]

    # для каждой переданной в сценарий строки
    for line in sys.stdin:
        # если она совпадает с регулярным выражением, записать ее в выходной поток stdout
        if re.search(regex, line):
            sys.stdout.write(line)


# In[ ]:



