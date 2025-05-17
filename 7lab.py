import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.naive_bayes
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df = pd.read_csv('lab7.csv',encoding='cp1251')
print(df.head(5))
#   выделение данных и меток признаков
y = df['gender'].astype(int)
X = df.drop('gender', axis=1)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y,
train_size=0.5, random_state=10)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
#   классификация наблюдений наивным байесовским методом, вывод верных и не верных результатов
gnb = sklearn.naive_bayes.GaussianNB()
y_v = list(y_valid)
y_pred = gnb.fit(X_train, y_train).predict(X_valid)
print('True: ', (y_v != y_pred).sum())
print('Not true: ', (y_v == y_pred).sum())
#   Классификация методом деревьев
DT = DecisionTreeClassifier(max_depth=4, random_state=10)
print(DT.fit(X_train, y_train))
plt.subplots(1,1,figsize = (10,10))
tree.plot_tree(DT, filled = True)
plt.show()