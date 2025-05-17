import sklearn as sk
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#   Первые 5 строк датасета
df = pd.read_csv('lab8.csv',encoding='cp1251')
print(df.head(5))
#   Стандартизация набора данных
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)
x_normal = normalize(x_scaled)
x_normal = pd.DataFrame(x_normal)
print(x_normal)
#   Реализация алгоритма анализа главных компонентов
pca = PCA(n_components=2)
x_principal = pca.fit_transform(x_normal)
x_principal = pd.DataFrame(x_principal)
x_principal.columns = ['V1', 'V2']
print(x_principal.head())
x_principal.to_csv('lab8new.csv', header=True)
#   Реализация алгоритма DBSCAN
df = pd.read_csv('lab8new.csv',encoding='cp1251')
dbscan = DBSCAN(eps=0.036, min_samples=4).fit(df)
labels = dbscan.labels_
df['cluster'] = dbscan.labels_
print(df.tail())
#   Вывод меток кластеров, количество кластеров, а также процент наблюдений, которые кластеризовать не удалось
print(set(dbscan.labels_))
print(len(set(dbscan.labels_)) - 1)
print(list(dbscan.labels_).count(-1) / len(list(dbscan.labels_)))
#   Визуализация полученных данных
plt.figure(figsize=(10, 8))
plt.scatter(df['V1'], df['V2'])
plt.title("Implementation of DBSCAN Clustering", fontname="Times New Roman",fontweight="bold")
plt.show()