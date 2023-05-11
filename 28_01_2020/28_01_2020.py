import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv('28_01_2020/weather_features.csv')

# Calcolo le istanze nel dataset
number_of_instances = df.shape[0]
# print(number_of_instances)

# verifica valori null
missing_value = df.isnull().sum()
# print(missing_value)

# controlla se il target è bilanciato
# print(df['weather_main'].unique())
target = df['weather_main'].value_counts()
# print(target)
distr = target/target.sum()
# print(distr)

# Le rilevazioni con pressione e umidità uguale a 0 sono irreali. Quante sono queste rilevazioni? Eliminarle dal dataset
press = df['pressure'].value_counts()[0]
hum = df['humidity'].value_counts()[0]
# print(press.sum(), hum.sum())
df = df.drop(df[df.loc[:,'pressure'] == 0].index)
df = df.drop(df[df.loc[:,'humidity'] == 0].index)
a = df.loc[:,'pressure'] == 0
b = df.loc[:,'humidity'] == 0
# print(a.sum(), b.sum())

# Analizzare la temperatura massima rilevata. Valutare se la distribuzione dei valori assume un 
# andamento simile a una gaussiana. Considerare poi le rilevazioni che si collocano all’interno del 
# 5% delle temperature più alte.  Le città sono equamente presenti in quella fascia di rilevazioni? 
# Come è il tempo complessivo nei giorni in cui la temperatura massima è in quella fascia per ogni 
# città?
# print(df['temp'].max())
# sns.histplot(data=df['temp'], kde=False)
# plt.show()
# sns.kdeplot(data=df['temp'], fill=True)
# plt.show()
# print(df['temp'].mean())
# print(df['temp'].var())
percent = number_of_instances * 0.05
sort = df.sort_values(by='temp', ascending=False)[:int(percent)]
city_distr = sort['city_name'].value_counts()
cities = sort['city_name'].unique()
freq = city_distr/city_distr.sum()
# print(freq)
# print(sort['weather_main'].value_counts())

# Verificare  se  quando  nevica  la  temperatura  sia  prossima  alla  temperatura  di  congelamento
freezing_temp = 273.15
temp = df[df['weather_main'] == 'snow']

# print((temp['temp'] <= freezing_temp).sum())

# Confrontare  l’escursione  termica  media  (temp_max-temp_min)  registrata  nei  giorni  in  cui
# nevica, con quella delle giornate che sono  all’interno del 5% delle temperature più alte
msnow = (temp['temp_max']-temp['temp_min']).mean()
mhot = sort['temp_max']-sort['temp_min']
# print('Giornate neve: ', msnow)
# print('Giornate calde: ', mhot.mean())

df.drop(["dt_iso","city_name","weather_description", "weather_icon","weather_id", "clouds_all"], axis=1, inplace=True)
encoder = preprocessing.OneHotEncoder()
encoder.fit(df[['weather_main']])
encoder_df = pd.DataFrame(encoder.transform(df[['weather_main']]).toarray())
# one_hot_encoded_data = pd.get_dummies(df, columns = ['weather_main'])
df = df.join(encoder_df)
df.drop(['weather_main'],axis=1, inplace=True)
df = df.dropna()

y = df.drop(['temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h'], axis=1)
X = df.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=1/3)
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

test_prediction = tree.predict(X_test)
train_prediction = tree.predict(X_train)
# print(accuracy_score(y_test, test_prediction), accuracy_score(y_train, train_prediction))

'''logreg = LogisticRegression(random_state=0, multi_class='ovr', solver='saga')
logreg.fit(X_train, y_train)

prediction1 = logreg.predict(X_test)
print(accuracy_score(y_test, prediction1))'''

# Confrontare l’accuratezza ottenuta nel punto precedente con l’accuratezza che si ottiene con un una 10 Fold cross validation
scores = cross_val_score(tree, X, y, cv=10)
# print(scores)

norm = preprocessing.Normalizer()
Xnorm = norm.fit_transform(X, y)
# min_max_scaler = preprocessing.MinMaxScaler()
# Xnorm = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xnorm, y , test_size=1/3)
tree.fit(X_train, y_train)

test_prediction = tree.predict(X_test)
train_prediction = tree.predict(X_train)
# print(accuracy_score(y_test, test_prediction), accuracy_score(y_train, train_prediction))

