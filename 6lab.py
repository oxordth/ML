import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from math import isnan
#   первые 5 строк датасета
df = pd.read_csv('otchet6.csv',encoding='cp1251')
print(df.head(5))
#   Переформатируем данные
np_df = df.to_numpy()
np_df = [[elem for elem in row[1:] if isinstance(elem,str)] for row in
np_df]
# print(np_df)
#   Уникальные продукты
np_df = df.to_numpy()
np_df = [[elem for elem in row[4:] if isinstance(elem,str)] for row in
np_df]
unique_items = set()
for row in np_df:
 for elem in row:
    unique_items.add(elem)
# print(unique_items)
#   Нужный формат данных
te = TransactionEncoder()
te_ary = te.fit(np_df).transform(np_df)
df_new = pd.DataFrame(te_ary, columns=te.columns_)
# print(df_new)
#   Алгоритм FPGrowth
te = TransactionEncoder()
te_ary = te.fit(np_df).transform(np_df)
df_new = pd.DataFrame(te_ary, columns=te.columns_)
fpg = fpgrowth(df_new, min_support=0.03, use_colnames = True)
# print(fpg)
#   График
min_support_range = np.arange(0.01, 0.1, 0.01)
itemsets_lengths = []
threshold_supports = []
threshold_lengths = []
last_itemset_len = len(df_new.columns)
for min_support in min_support_range:
 fpg = fpgrowth(df_new, min_support=min_support, use_colnames=True)
 itemsets_lengths.append(len(fpg))
 fpg['length'] = fpg['itemsets'].apply(lambda x: len(x))
 current_itemset_max_len = fpg['length' ].max()
 if isnan(current_itemset_max_len):
    current_itemset_max_len = 0
 if current_itemset_max_len < last_itemset_len:
    last_itemset_len = current_itemset_max_len
 threshold_supports.append(min_support)
 threshold_lengths.append(len(fpg))
plt.figure()
plt.plot(min_support_range.tolist(), itemsets_lengths)
plt.plot(threshold_supports, threshold_lengths, 'ro')
plt.show()