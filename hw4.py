# -*- coding: utf-8 -*-
"""hw4

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EN67HgEPBkQjQHXg2HPWqgeof51HLbo-
"""

#Задача 1.Вводится строка. Нужно превратить строку в список, разбив строку на слова и вывести список из слов, записанный в обратном порядке.
#Ввод:Death there mirth way the noisy merit
#Вывод:['merit', 'noisy', 'the', 'way', 'mirth', 'there', 'Death']

# ввод текста строки
text_in = input('Введите текст: ')

#разделения строки на слова  в данном случае разделитель - пробел(можно поставить любой)
text_in_list = text_in.split()

#изменения порядка элементов в списке на противоположный
text_in_list.reverse()

print(f'перевертнутый список: {text_in_list}')

#Задача 2. Вводится строка из не менее 15 символов, программа выводит на экран символы с чётными номерами (нумерация с 0).
#Ввод: In the hole in the ground there lived a hobbit
#Вывод: I h oei h rudteelvdahbi

# ввод текста строки
text_in = input('Введите текст: ')

# срез от 0 до конца с шагом 2
print(text_in[::2])

#Задача 3.Вводится список чисел через пробел и натуральное число n - степень. Нужно возвести в заданную степень n все введенные числа.
#Желательно сделать, используя списочные выражения
#Ввод:3 5 -7 -13 43 8 0 -13 8 -1 2
#Вывод: [9, 25, 49, 169, 1849, 64, 0, 169, 64, 1]
# ввод текста строки и добавление к списку метод split()
sim = input("Введите числа через пробел: ").split()
# ввод степени
n = int(input("Введите степень"))
#цикл для преобразования каждого значения в списке из строки в чило и возведение его в степень
for i in range(len(sim)):
    sim[i] = int(sim[i])**n

print(sim)

#Задача 4. Вводится строка с текстом и символ. Требуется удвоить вхождение введённого символа в текст.
#Текст состоит из слов, записанных латинскими буквами через пробел, знаков препинания.
#Подсказка: вспомните, как работает метод  replace

nums = input("Введите числа через пробел: ").split()
#цикл для преобразования каждого значения в списке из строки в чило и возведение его в степень
for i in range(len(nums)):
  if nums[i].isdigit():
    nums[i] = int(nums[i])*2

print(nums)

# Ввод строки с текстом
text = input("Введите текст: ")

# Ввод символа, который нужно удвоить
symbol = input("Введите символ для удвоения: ")

# Используем метод replace для замены символа в тексте
doubl_text = text.replace(symbol, symbol * 2)

# Вывод результата
print("Результат:", doubl_text)

#Задача 5. Дана строка. Программа подсчитывает количество символов x и y и выводит строку вида "x: (число), y: (число)."
#Подсказка: вспомните метод count, изученный на занятии

simb = input().split()
count_x = simb.count('x')
count_y = simb.count('y')
print(f'x: {count_x}, y: {count_y}')

#Задача 6.Вводится текст со сбалансированными скобками, программа выводит на экран текст без скобок и их содержимого. На пробелы и знаки препинания внимание не обращать, вложенных скобок в исходной строке нет.
#Подсказка: вспомните метод find, изученный на занятии.
#Ввод:When he saw Sally (a girl he used to go to school with) in the shop, he could not believe his eyes. She was fantastic (as always)!
#Вывод: a girl he used to go to school with as always
txt = input()
str1 = txt[txt.find('(')+1:txt.find(')')] + ' ' + txt[txt.rfind('(')+1:txt.rfind(')')]

print(str1)