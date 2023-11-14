# -*- coding: utf-8 -*-
"""hw7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1malYDhzwj3Yr0ZycbG8hD5QseTEchVOh
"""

#Задача 1. Напишите программу на Python для поочередного сложения элементов двух заданных списков, используя map и lambda.
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]

sum_of_lists = list(map(lambda x, y: x + y, list1, list2))
print(sum_of_lists)

#Задача 2. Напишите программу на Python для поиска чисел из списка, кратных девятнадцати или тринадцати, используя filter и  Lambda.

list_kr = [1, 2, 3, 4, 19, 13]

kr_numbers = list(filter(lambda x: x % 19 == 0 or x % 13 == 0, list_kr))

if kr_numbers:
  print("Числа, кратные девятнадцати или тринадцати:", *kr_numbers)
else:
  print('таких нет')

#Задача 3. Напишите программу на Python для вычисления наибольшего элемента в списке при помощи reduce

from functools import reduce
seq1 = [5, 7, 11, 8, 9]
max_element = reduce(lambda a, b: a if a > b else b, seq1)
print(max_element)

#Домашнее задание Типы данных. поправил код, первоначально неправильно понял задание.

#Задача 6.Вводится текст со сбалансированными скобками, программа выводит на экран текст без скобок и их содержимого. На пробелы и знаки препинания внимание не обращать, вложенных скобок в исходной строке нет.
#Подсказка: вспомните метод find, изученный на занятии.
#Ввод:When he saw Sally (a girl he used to go to school with) in the shop, he could not believe his eyes. She was fantastic (as always)!
#Вывод: a girl he used to go to school with as always

txt = input()
result = ""

while '(' in txt:
    start_index = txt.find('(')
    end_index = txt.find(')')
    result += txt[start_index + 1:end_index] + ' '
    txt = txt[:start_index] + txt[end_index + 1:]

print(result.strip())

