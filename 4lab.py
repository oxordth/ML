import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import MDS
import seaborn as sns
import umap
# Сводная информация о выбранном датасете
df = pd.read_csv('Mall_Customers.csv',encoding='cp1251')
print(df.info())
# Деление данных на признаки и классы
variables = ['Age', 'Annual_Income', 'Spending_Score']
x = df[variables].values

# Масштабирование данных
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Применение PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(x_scaled)
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
print(pca_df)
# Добавление пола (класса) к данным
pca_df['Gender'] = df['Gender']

# Диаграмма рассеяния для PCA
# sns.scatterplot(x='PC1', y='PC2', hue='Gender', data=pca_df)
# plt.title("PCA")
# plt.show()
#   Реализация ICA
variables = ['Age', 'Annual_Income', 'Spending_Score']
x = df[variables].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
ica = FastICA(n_components=2)
ica_data = ica.fit_transform(x_scaled)
ica_df = pd.DataFrame(data=ica_data, columns=['IC1', 'IC2'])
print(ica_df)
ica_df['Gender'] = df['Gender']
#   Реализация UMAP
variables = ['Age', 'Annual_Income', 'Spending_Score']
x = df[variables].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
umap_ = umap.UMAP(n_components=2)
umap_data = umap_.fit_transform(x_scaled)
umap_df = pd.DataFrame(data=umap_data, columns=['UM1', 'UM2'])
umap_df['Gender'] = df['Gender']
#   Реализация MDS
variables = ['Age', 'Annual_Income', 'Spending_Score']
x = df[variables].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
mds = MDS(n_components=2)
mds_data = mds.fit_transform(x_scaled)
mds_df = pd.DataFrame(data=mds_data, columns=['MD1', 'MD2'])
mds_df['Gender'] = df['Gender']
#   Графики MDS и UMAP
plt.figure(figsize=(8, 4))
plt.subplot(2, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue=df['Gender'], data=pca_df)
plt.title("PCA")
plt.subplot(2, 2, 2)
sns.scatterplot(x='IC1', y='IC2', hue=df['Gender'], data=ica_df)
plt.title("ICA")
plt.subplot(2, 2, 3)
sns.scatterplot(x='MD1', y='MD2', hue=df['Gender'], data=mds_df)
plt.title("MDS")
plt.subplot(2, 2, 4)
sns.scatterplot(x='UM1', y='UM2', hue=df['Gender'], data=umap_df)
plt.title("UMAP")
plt.show()