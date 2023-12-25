# -*- coding: utf-8 -*-
"""hw9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wf4ln6FM2sMcEqMYqraRmV0tJEX6QKf4

"""

Задание 1:
Напишите программу, которая будет находить наименьшее общее кратное двух чисел.
Ввод: два числа a и b
Вывод: наименьшее общее кратное чисел a и b


Задание 2:
Напишите программу, которая будет находить все простые числа от 1 до n.
Ввод: число n
Вывод: список простых чисел от 1 до n


Задание 3:
Напишите программу, которая будет находить все делители числа n.
Ввод: число n

#Задание 1:
def min_multiple_count(a,b):
  from math import gcd
  lcm = (a * b) // gcd(a, b)
  return lcm

a,b = int(input()),int(input())
print(f'наименьшее общее кратное чисел {a} и {b} - {min_multiple_count(a,b)}')

#Задание 2:
def all_simple_nums(n):
    simp_num = []
    for num in range(2, n + 1):
        is_simp_num = True
        for i in range(2, num):
            if num % i == 0:
                is_simp_num = False
                break
        if is_simp_num:
            simp_num.append(num)
    return simp_num

n = int(input())
print('список простых чисел от 1 до', n, '-', *all_simple_nums(n))

#Задание 3:
def all_devide_n(s):
  devide_s = []
  for i in range(1,s + 1):
    if s%i == 0:
      devide_s.append(i)
  return devide_s

s = int(input())
print('все делители числа ', s ,'-', *all_devide_n(s))