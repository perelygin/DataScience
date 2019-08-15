# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:31:02 2019

@author: perelygin
"""
#функции
def in_dict(value,dct):  # проверка наличия ключа в словаре
    if value in dct:
        return value +" Есть такое"
    else:
        return value +" Нет такого"
#словари
Dict1 = {"Masha":10,"Sasha":14,"Dasha":54}
print(Dict1["Masha"]) #доступ по ключу
try:
   print(Dict1["pipa"]) #доступ по несуществующему ключу
except KeyError:
   print("We don't have this key") 
   

print(in_dict("Masha",Dict1))
print(in_dict("Mashunia",Dict1))

print(Dict1.get("Masha22",-1)) #

Dict1["Koza"]=90  #присвоение нового значения
print(Dict1)
if "Koza1" in Dict1.keys(): #ключи словаря
    print("есть такой ключ") 
else:
    print("нет такой ключ") 
    
document = {"черный","черный","черный","синий","синий","зеленый","зеленый","зеленый","зеленый"}
color_count={}
for color in document:
    print(color)
    if color in color_count:
       color_count[color] += 1
    else:
       color_count[color] = 1

print(color_count)