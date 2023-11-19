# -*- coding: utf-8 -*-
"""hw5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aZ25EF-maSykkWilQiyE_baDyEGHKkrg

Задача 1. Нужно написать программу, которая бы вычисляла количество городов, названия которых повторяются. В первой строке указывается количество городов n, затем вводятся n строк с названиями городов.
- Ввод:
- 6
- Курск
- Калуга
- Анапа
- Анапа
- Абинск
- Курск
- Вывод: 4
"""

num_of_cities = int(input('Введите количество городов: '))
count = 0
cities = []

for i in range(num_of_cities):
    city_name = input().lower()
    cities.append(city_name)

for city in cities:
    if cities.count(city) >= 2:
        count += 1

print(count)

""" Задача 2.Младшая сестра Фёдора - Соня пишет сочинение по литературе и отправляет файлы учительнице.
Фёдор знает, что Соня никогда не ставит заглавные буквы, так как для набора текста использует компьютер. Пока никто не видит, Фёдор решил внести исправления в сочинение сестры и написал программу, которая восстанавливает заглавные буквы в строке.
Программа работает по следующему алгоритму:
* сделать заглавной первую букву в строке, не считая пробелы;
* сделать заглавной первую букву после точки, восклицательного или вопросительного знака, не считая пробелы.
Ввод: на этом заканчиваю свое сочинение. поставьте пятерку, Мария Ивановна! я очень старалась!
Вывод: На этом заканчиваю свое сочинение. Поставьте пятерку, Мария Ивановна! Я очень старалась!
"""

#Ввод строки
string = input()
#заглавная первая буква
string = string[0].upper() + string[1:]
#цикл для поиска значений в тексте  ('.', '!', '?')
for i in range(len(string)):
    if (string[i] in '.!?') and (i + 2 < len(string)):
      string = string[:i+2] + string[i+2].upper() + string[i+3:]

print(string)

"""Задача 3.Проверка на анаграмму. Пользователь вводит две строки. Напишите программу, которая определяет, являются ли эти строки анаграммами (содержат одни и те же символы, но в разном порядке). Выведите соответствующее сообщение.
Подсказка: воспользуйтесь множеством для сравнения уникальных символов в строке, но не забудьте избавиться от пробелов.

"""

string1 = input().replace(' ','').lower()
string2 = input().replace(' ','').lower()

string1 = set(string1)
string2 = set(string2)

if string1 == string2:
  print('анаграмма')
else:
  print('не анаграмма')

"""Задача 4. В олимпиаде по математике школьникам было предложено решить одну задачу по алгебре, одну по геометрии и одну по тригонометрии. Напишите программу, которая определяет, сколько учащихся решили все задачи.
Формат ввода:
На вход подается три строки:
	•	в первой строке указаны фамилии школьников, решивших задачу по алгебре;
	•	во второй строке указаны фамилии школьников, решивших задачу по геометрии;
	•	в третьей строке указаны фамилии школьников, которые решили задачу по тригонометрии.
Формат вывода:
Требуется вывести в алфавитном порядке через пробел фамилии учащихся, решивших все три задачи олимпиады. Если таких нет, вывести строку "Все три задачи никто не решил".
Ввод:
Иванов Петров Сидоров Михайлов
Иванов Михайлов
Сидоров Михайлов
Вывод: Михайлов

"""

alg = set(input().split())
geo = set(input().split())
tri = set(input().split())
three = (alg & geo & tri)

if three:
  print(*sorted(three))
else:
  print('Все три задачи никто не решил')
