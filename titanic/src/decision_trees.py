# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:15:34 2017

@author: thor
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import Imputer, LabelEncoder


#%%

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#%%

discrete_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
continuous_features = ['Age', 'Fare']

#%%

train = train[discrete_features + continuous_features + ['Survived']]
train = train.dropna(axis = 0, how = 'any')

test = test[discrete_features + continuous_features]
test = test.dropna(axis = 0, how = 'any')

# TODO: impute missing values instead of removing NAs


#%% GET DUMMIES
# 'dropFirst=False', because multicollinearity okay for decision trees

train = pd.get_dummies(train, drop_first=False)


#%%

x_train = train.drop('Survived', axis=1)
y_train = train['Survived']

x_test = test


#%% EVALUATE MODEL ACCURACY

model = RandomForestClassifier(random_state=0, verbose=0, n_estimators=1000)
cross_val_score(model, x_train, y_train, cv=5, n_jobs=-1)


#%%

model.fit(X=x_train, y=y_train.values)

y_pred = model.predict(x_test)


#%% TODO: impute values, to avoid losing rows


