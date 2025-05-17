import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv',encoding='cp1251')
print(df.head(5))
print("\n")

# Зависимость оценки расходов от дохода покупателя
plt.hist(df['Spending_Score'], bins=30, color='black', alpha=0.8)
plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Spending score based on income')
plt.show()

#Коэффициент корелляции расходов от доходов
correlation = df['Spending_Score'].corr(df['Annual_Income'])
print("correlation between spending score and income:", correlation)
print("\n")

#Выбросы данных в столбце доходов
plt.boxplot(df['Annual_Income'])
plt.ylabel('Income')
plt.title('Income emissions analysis')
plt.show()

#График возрастов покупателей
s_r = df['Age'].value_counts()
s_r.plot(kind='bar')
plt.xlabel('age')
plt.ylabel('id')
plt.title('Ages among buyers')
plt.show()

#Результаты разведочного анализа в общем дешборде
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.hist(df['Spending_Score'], bins=30, color='black', alpha=0.8)
plt.xlabel('income')
plt.ylabel('spending score')
plt.title('spending score based on income')
plt.subplot(2, 2, 2)
plt.scatter(df['Age'], df['Annual_Income'], color='black')
plt.xlabel('age')
plt.ylabel('income')
plt.title('scatterplot between age and income')
plt.subplot(2, 2, 3)
plt.boxplot(df['Annual_Income'])
plt.ylabel('income')
plt.title('income emissions analysis')
sns.heatmap(df.corr(), annot=True, fmt='.2g')
plt.title('heat map')
plt.tight_layout()
plt.show()
