# -*- coding: utf-8 -*-
"""hw6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EAB68n3WjBRBYqJepzAhmuSBhsm_Y6yS
"""

#Напишите функцию, которая берёт на вход строку и возвращает true, если
#она является палиндромом и false в противном случае.

def palindrom(n):
  n = n.replace(' ', '')
  n = n.lower()
  if n == n[::-1]:
    return True
  else:
    return False

palindrom(input())

#Напишите и вызовете для себя или какого-нибудь персонажа функцию,
#которая берёт на вход имя, фамилию, отчество и возраст и возвращает
#строку вида “Иванов Иван Иванович 1973 г.р. зарегистрирован”

def registration(surname, name,surname_2,age):
  print(f'{surname} {name} {surname_2} {age}  г.р. зарегистрирован')

registration(input('введите фамилию: '),input('введите имя: '),input('введите отчество: '),input('введите возраст: '))

#Напишите функцию, которая берёт на вход 2 или 3 натуральных числа
#и возвращает их максимум. Встроенным методом max()
#пользоваться нельзя Возможно, вам потребуется указать аргумент по умолчанию.
def max_number(*nums):
  if len(nums) == 0:
    return 'числа не введены'
  num_max = nums[0]
  for i in range(1,len(nums)):
    if num_max < nums[i]:
      num_max = nums[i]
  print(f'максимальное введенное число: {num_max}')

max_number(1,20,3,7,9,6)