# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:34:48 2017

@author: thor
"""

#%% PACKAGES

import pandas as pd
from ggplot import *
import scipy as sp
import numpy as np
from scipy import stats

#%% LOAD DATA

train = pd.read_csv('../data/train.csv')

#%% CHI2-TESTS FOR FEATURES

discrete_features = ['Pclass', 'Sex', 'SibSp',
            'Parch', 'Embarked']

def feature_chi2(feature_label):
    ct = pd.crosstab(train['Survived'], train[feature_label], margins=True)
    return sp.stats.chi2_contingency(ct)[1]

print('Chi2 values for discrete variables:')
for feature in discrete_features:
    chi2 = feature_chi2(feature)
    print(str(feature) + ': ' + str(chi2))

#%% BOXPLOTS AND INDEPENDENCE TESTS FOR CONTINUOUS FEATURES

continuous_features = ['Age', 'Fare']

ggplot(train, aes(x='Survived', y='Age')) + geom_boxplot()
ggplot(train, aes(x='Survived', y='Fare')) + geom_boxplot()

alive = train[(train.Survived == 1)]
dead = train[(train.Survived == 1)]

# Welch's t-test
stats.ttest_ind(alive['Age'], dead['Age'], nan_policy='omit', equal_var=False)
stats.ttest_ind(alive['Fare'], dead['Fare'], nan_policy='omit', 
                equal_var=False)

# non-parametric 'ANOVA'
stats.kruskal(alive['Age'], dead['Age'], nan_policy='omit')
stats.kruskal(alive['Fare'], dead['Fare'], nan_policy='omit')

print('For features AGE and FARE (price):')
print('Null-hypothesis of equal averages could not be rejected.')

#%% SPLIT DATA IN X AND Y

X = train[discrete_features + continuous_features]
Y = train['Survived']

#%% TRAIN DECISION TREES

# TODO

#%% RULE EXTRACTION FROM TREES

# TODO

#%% CHECKING FOR MULTICOLLINEARITY IN FEATURES

# TODO

#%% LOGISTIC REGRESSION

# TODO



