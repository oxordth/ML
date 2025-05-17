import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.naive_bayes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Загрузка данных и вывод информации о таблицах и первых 5 строк датасета
print("--------------------------Предобработка данных--------------------------")
df = pd.read_csv('trainset.csv')
print(df.info())
print(df.head(5))
# Анализ выбросов, вывод дублированных записей и их удаление
print("Кол-во дубликатов в датасете: ", len(df[df.duplicated()]))
df = df.drop_duplicates()
print("Кол-во дубликатов в датасете: ",len(df[df.duplicated()]))
df = df.drop(['playerId', 'Name'], axis = 1)
df = df.dropna()
print("Названия колонок датасета: ", list(df.columns))
# # Преобразование категориальных переменных и вывод информации о таблицах и первых 5 строк датасета после преобразования
df_sex = LabelEncoder().fit_transform(df['Sex'].values)
df_eq = LabelEncoder().fit_transform(df['Equipment'].values)
df['Sex'] = df_sex
df['Equipment'] = df_eq
print(df.info())
print(df.head(5))
# Понижение размерности данных методом главных компонент
print("--------------------Стандартизация данных-------------------")
scaler = PCA()
# Удаление ненужного столбца
names = df.columns
d = scaler.fit_transform(df)
scal_df = pd.DataFrame(d, columns=names)
print(scal_df.head(10))
df_new = df.drop(['BestBenchKg'], axis = 1)
names = df_new.columns
b = scaler.fit_transform(df_new)
scal_df = pd.DataFrame(b, columns=names)
print(scal_df.head(10))
# Разведочный анализ данных 
#   ящик с усами для экипировки
plt.boxplot(df['Equipment'])
plt.ylabel('Equipment')
plt.show()
#   ящик с усами для колонки возраста
plt.boxplot(df['Age'])
plt.ylabel('Age')
plt.show()
# тепловая карта датасета
sns.heatmap(df.corr(), annot=True, fmt='.2g')
plt.title('heat map')
plt.tight_layout()
plt.show()
# Распределение частоты каждого вида экипировки
s_r = df['Equipment'].value_counts(sort = True)
s_r.plot(kind = 'bar')
plt.xlabel("Тип экипировки (0 - комбинезон многослойный, 1 - без экипировки, 2 - комбинезон однослойный, 3 - бинты)")
plt.ylabel("Количество спортсменов")
plt.show()

# классификация наблюдений наивным байесовским методом
print("--------------------Классификация данных-----------------------")
y = pd.cut(df['BestBenchKg'], bins = 5, labels = False)
X = b
gnb = sklearn.naive_bayes.GaussianNB()
kn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
gt = []
gf = []
kt = []
kf = []
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, random_state=10)
y_v = y_test
y_pred_g = gnb.fit(X_train, y_train).predict(X_test)
y_pred_k = kn.fit(X_train, y_train).predict(X_test)
gt.append((y_v == y_pred_g).sum())
gf.append((y_v != y_pred_g).sum())
kt.append((y_v == y_pred_k).sum())
kf.append((y_v != y_pred_k).sum())
print ("Метод k-ближних соседей\nВерных наблюдений: ", (y_v==y_pred_k).sum(), "\nНеверных наблюдений", (y_v!=y_pred_k).sum())
print ("Наивный байесовский метод\nВерных наблюдений: ", (y_v==y_pred_g).sum(), "\nНеверных наблюдений", (y_v!=y_pred_g).sum())
plt.bar("TrueB", sum(gt))
plt.bar("FalseB", sum(gf))
plt.bar("TrueK", sum(kt))
plt.bar("FalseK", sum(kf))
plt.xlabel('Методы(слева - Байес, справа - k-ближних соседей)')
plt.ylabel('Количество наблюдений')
plt.title('Сравнение верных и неверных наблюдений')
plt.legend()
plt.show()

# Обучение модели на обучающей выборке
y = df['BestBenchKg']
X = pd.DataFrame(b)
model = RandomForestRegressor(n_estimators=100, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y,
train_size=0.95, random_state=10)
model.fit(X_train, y_train)
def bench_kg(Sex,Equipment,Age,BodyweightKg,BestSquatKg,BestDeadliftKg):
    features = np.array([Sex,Equipment,Age,BodyweightKg,BestSquatKg,BestDeadliftKg]).reshape(1, -1)
    features = scaler.transform(features) # Нормализация входных данных
    # Предсказание на основе входных данных
    bench_prediction = model.predict(features)
    return bench_prediction[0]

def prediction():
    while True:
        op = int(input("1. Начать проверку\n2. Выход из программы\nВведите число: "))
        if op == 1:
            print("Параметры спортсмена:\n")
            sex= int(input("Введите пол (женщина - 0, мужчина - 1): "))
            eq=int(input("Введите экипировку (0 - комбинезон многослойный, 1 - без экипировки, 2 - комбинезон однослойный , 3 - бинты): "))
            age=float(input("Введите возраст: "))
            weight=float(input("Введите вес (кг): "))
            squat=float(input("Введите вес на снаряде для приседаний: "))
            deadlift=float(input("Введите вес на снаряде для становой тяги: "))
            result = bench_kg(sex, eq, age,weight, squat, deadlift)
            print("Прогнозируемый Вес снаряда:", result)
        else:
            return
# prediction()
# Спортсмен-мужчина без экипировки
print("Параметры спортсмена:")
print("Пол (женщина - 0, мужчина - 1): 1")
print("Экипировку (0 - комбинезон многослойный, 1 - без экипировки, 2 - комбинезон однослойный , 3 - бинты): 1")
print("Возраст: 23")
print("Вес (кг): 87.3")
print("Вес на снаряде для приседаний (кг): 205")
print("Вес на снаряде для становой тяги (кг): 235")
print("Прогнозируемый вес снаряда: ~", round(bench_kg(1, 1,23.0,87.3,205.0,235.0)))
print("Ожидаемый результат: ~125\n")
# Спортсмен-мужчина с экипировкой (бинты)
print("Параметры спортсмена:")
print("Пол (женщина - 0, мужчина - 1): 1")
print("Экипировку (0 - комбинезон многослойный, 1 - без экипировки, 2 - комбинезон однослойный , 3 - бинты): 3")
print("Возраст: 23")
print("Вес (кг): 73.48")
print("Вес на снаряде для приседаний (кг): 220")
print("Вес на снаряде для становой тяги (кг): 260")
print("Прогнозируемый вес снаряда: ~", round(bench_kg(1,3,23.0,73.48,220.0,260.0)))
print("Ожидаемый результат: ~157.5\n")
# Спортсмен-женщина с однослойным комбинезоном
print("Параметры спортсмена:")
print("Пол (женщина - 0, мужчина - 1): 0")
print("Экипировку (0 - комбинезон многослойный, 1 - без экипировки, 2 - комбинезон однослойный , 3 - бинты): 2")
print("Возраст: 19")
print("Вес (кг): 51")
print("Вес на снаряде для приседаний (кг): 115")
print("Вес на снаряде для становой тяги (кг): 127.5")
print("Прогнозируемый вес снаряда: ~", round(bench_kg(0,2,19.5,51.2,115.0,127.5)))
print("Ожидаемый результат: ~62.5\n")
# Спортсмен-мужчина с многослойным комбинезоном
print("Параметры спортсмена:")
print("Пол (женщина - 0, мужчина - 1): 1")
print("Экипировку (0 - комбинезон многослойный, 1 - без экипировки, 2 - комбинезон однослойный , 3 - бинты): 0")
print("Возраст: 28")
print("Вес (кг): 75")
print("Вес на снаряде для приседаний (кг): 322")
print("Вес на снаряде для становой тяги (кг): 250")
print("Прогнозируемый вес снаряда: ~", round(bench_kg(1,0,28.0,74.93,-322.5,250.0)))
print("Ожидаемый результат: ~185\n")