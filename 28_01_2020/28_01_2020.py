import pandas 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pandas.read_csv('../weather_features.csv')

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

print((temp['temp'] <= freezing_temp).sum())