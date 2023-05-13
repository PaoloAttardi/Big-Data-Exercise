import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, pair_confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('08_01_2020/train.csv')
# print(df.shape[0])
# print(df.isna().sum())
# df['price_range'].value_counts().plot.bar()
# print(df['price_range'].value_counts())
# plt.show()

# print(df['sc_w'].describe)
# sns.lineplot(data=df, x='price_range', y='sc_w')
df.drop(df[df.loc[:,'sc_w'] == 0].index, inplace=True)
# print(df.groupby(['price_range']).mean()['sc_w'])
# plt.show()

# print(df.groupby(['price_range']).mean()['battery_power'])
for i in range(4):
    histdf = df.loc[df['price_range'] == i]
    #sns.histplot(histdf, x='battery_power')
    # plt.title(f'Price Range {i}')
    # plt.show()

fourgdf = df.loc[df['four_g'] == 1]
# print(fourgdf['three_g'].unique())

# print(fourgdf['blue'].value_counts())
# print(fourgdf['wifi'].value_counts())


y = df['price_range']
X = df.drop('price_range', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=1/4, stratify=y)
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

test_pred = tree.predict(X_test)
train_pred = tree.predict(X_train)
# print(accuracy_score(y_test, test_pred), accuracy_score(y_train, train_pred))

'''logreg = LogisticRegression()
logreg.fit(X_train, y_train)

test_pred1 = logreg.predict(X_test)
train_pred1 = logreg.predict(X_train)'''
# print(accuracy_score(y_test, test_pred1), accuracy_score(y_train, train_pred1))
# sns.heatmap(confusion_matrix(y_pred=test_pred, y_true=y_test), annot=True, cmap='Blues')
# plt.show()
# sns.heatmap(confusion_matrix(y_pred=test_pred1, y_true=y_test), annot=True, cmap='Blues')
# plt.show()

scores = cross_val_score(tree, X, y, cv=10)
# scores2 = cross_val_score(logreg, X, y, cv=10)
# print(scores.mean(), scores2.mean())

param_grid = [{'criterion' : ['gini', 'entropy'],
              'max_features' : ['sqrt','log2'],
                'min_samples_split' : [2, 3, 4, 5]}]
scores = ['accuracy']
gridsearch = GridSearchCV(tree,param_grid,cv=10)
gridsearch.fit(X_train,y_train)
# print(gridsearch.best_params_)
# print(accuracy_score(y_test,gridsearch.predict(X_test)))

scaler = MaxAbsScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
tree.fit(X_train_sc, y_train)

test_pred = tree.predict(X_test_sc)
train_pred = tree.predict(X_train_sc)
# print(accuracy_score(y_test, test_pred), accuracy_score(y_train, train_pred))

X_train_bin, X_test_bin = X_train, X_test
X_train_bin['ram'] = pd.cut(X_train['ram'], bins=4, labels=False)
X_test_bin['ram'] = pd.cut(X_test['ram'], bins=4, labels=False)
tree.fit(X_train_bin, y_train)

test_pred = tree.predict(X_test_bin)
train_pred = tree.predict(X_train_bin)
print(accuracy_score(y_test, test_pred), accuracy_score(y_train, train_pred))