# -*- coding: utf-8 -*-
"""hw-2-3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OeQ3XwwiesUtmgNE0Q4EcbsS9CvNtFnf

Прочитайте представленный датасет(https://drive.google.com/drive/folders/1phOzEdwpLqxskFjRJSD-ZLRLk5Gh7DCz)(Подсказка: датасет надо сначала сохранить, а потом прочитать с помощию функции  read_csv)
"""

import pandas as pd

# Прочитать датасет из CSV файла
df = pd.read_csv('Customers.csv', sep=';')

print(df)

#Найдите людей, у которых возраст больше 30 и доход меньше 30000
df[(df['Age'] > 30) & (df['Income'] < 30000)]

# Найти юристов с опытом работы более 5 лет
df[(df['Profession'] == 'Lawyer') & (df['Work Experience'] > 5)]