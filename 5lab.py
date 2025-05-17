import pandas as pd
import numpy as np
import nltk
import pymorphy2
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from mlxtend.preprocessing import TransactionEncoder
from wordcloud import WordCloud
import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 
import csv
# Первые 5 строк датасета
df = pd.read_csv('lab5.csv', encoding='cp1251')
print(df.head(5))
# # Количество уникальных покупателей и количество уникальных продуктов
# quantity_id = list(set(df['id']))
# quantity_product = list(set(df['product']))
# print(len(quantity_id), len(quantity_product))
# Все товары каждого покупателя в одном списке
quantity_id = list(set(df['id']))
quantity_product = list(set(df['product']))
all_product = [[elem for elem in df[df['id'] == date]['product'] if elem in quantity_product] for date in quantity_id]
# print(all_product)
# # Товары в виде матрицы
te = TransactionEncoder()
te_allp = te.fit(all_product).transform(all_product)
df_matrix = pd.DataFrame(te_allp, columns=te.columns_)
# print(df)
# # Облако слов
# df = pd.read_csv('lab5.csv',encoding='cp1251')
# text = " ".join(df['product'])
# cloud = WordCloud().generate(text)
# plt.imshow(cloud)
# plt.axis('off')
# plt.show()
# # Избавление от стоп слов
# stop_words = stopwords.words('russian')
# df = pd.read_csv('lab5.csv',encoding='cp1251')
# text = " ".join(df['product'])
# cloud = WordCloud().generate(text)
# cloud = WordCloud(stopwords=stop_words).generate(text)
# plt.imshow(cloud)
# plt.axis('off')
# plt.show()
#   apriori
sup = 0.1
print(f'ИСХОДНЫЙ ДАТАСЕТ: Минимальный уровень поддержки = {sup}')
results = apriori(df_matrix, min_support=sup, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length']==2]
indexes = results.index.to_list()
results['id'] = [i for i in indexes]
print(results)
new_dataset = df[df['id'].isin(results['id'].tolist())][['date', 'id', 'product']]
new_dataset.to_csv('new_dataset.csv', index=False)
#   Открываем новый датасет и применяем apriori
df = pd.read_csv('new_dataset.csv', encoding='cp1251')
quantity_id = list(set(df['id']))
quantity_product = list(set(df['product']))
all_product = [[elem for elem in df[df['id'] == date]['product'] if elem in quantity_product] for date in quantity_id]
te = TransactionEncoder()
te_allp = te.fit(all_product).transform(all_product)
df_matrix = pd.DataFrame(te_allp, columns=te.columns_)
sup = 0.1
print(f'НОВЫЙ ДАТАСЕТ: Минимальный уровень поддержки = {sup}')
results = apriori(df_matrix, min_support=sup, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length']==2]
indexes = results.index.to_list()
results['id'] = [i for i in indexes]
print(results)
print(results['support'].max())

sup_range = np.arange(0.01, 0.5, 0.01)  # Диапазон значений для уровня поддержки
rules_counts = []  # Список для хранения количества правил на каждом уровне поддержки

for sup in sup_range:
    results = apriori(df_matrix, min_support=sup, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    results = results[results['length'] == 2]
    rules_counts.append(len(results))

# Определение максимального уровня поддержки, при котором перестают выводиться правила
max_support_level = sup_range[np.where(np.diff(rules_counts) == 0)[0][-1]]
max_rules_count = rules_counts[np.where(np.diff(rules_counts) == 0)[0][-1]]

# Определение минимального уровня поддержки для вывода максимального количества правил
min_support_level = sup_range[np.argmax(rules_counts)]
max_rules_count = max(rules_counts)

# Отображение на графике
plt.plot(sup_range, rules_counts)
plt.xlabel('Уровень поддержки')
plt.ylabel('Кол-во правил')
plt.title('Зависимость количества правил от уровня поддержки')
plt.axvline(x=max_support_level, color='r', linestyle='--', label=f'Мин. кол-во правил при УП {max_support_level:.2f}')
plt.axvline(x=min_support_level, color='g', linestyle='--', label=f'Макс. кол-во правил при УП {min_support_level:.2f}')
plt.legend()
plt.show()

print(f'Максимальный уровень поддержки: {max_support_level:.2f}')
print(f'Минимальный уровень поддержки для вывода максимального количества правил: {min_support_level:.2f}')