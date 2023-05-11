import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('08_01_2020/train.csv')
# print(df.shape[0])
# print(df.isna().sum())
# df['price_range'].value_counts().plot.bar()
# print(df['price_range'].value_counts())
# plt.show()

print(df['sc_w'].describe)
sns.lineplot(data=df, x='price_range', y='sc_w')
df.drop(df[df.loc[:,'sc_w'] == 0].index, inplace=True)
print(df.groupby(['price_range']).mean()['sc_w'])
plt.show()