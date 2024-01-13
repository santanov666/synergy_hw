# -*- coding: utf-8 -*-
"""hw-2-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1duD9I2vKCz4GTasrymWFkxhnIWpH2WLD

Создайте DataFrame, в котором будет не менее трех столбцов.
Выведите первые и последние 3 строки вашей таблицы
Сохраните ваш DataFrame в csv формат
"""

import pandas as pd

datafr = {
    'Имя': ['Саша', 'Надя', 'Виктор', 'Маша', 'Костя', 'Лида'],
    'Возраст': [25, 30, 28, 24, 29, 30],
    'рост': [175, 154, 160, 163, 210, 159]
}


df = pd.DataFrame(datafr)

# Вывод DataFrame
print(df)

# Вывод первых 3 строк DataFrame
print(df.head(3))

# Вывод последних 3 строк DataFrame
print(df.tail(3))

# Сохранение DataFrame в CSV
df.to_csv('lookitmyhorse_data.csv')