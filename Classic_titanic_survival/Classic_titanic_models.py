# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:48:24 2021

@author: kqll413
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Users\\kqll413\\Box Sync\\Data\\Hackathons\\Titanic')

df = pd.read_csv('titanic_train.csv')

df.columns

df.describe(include = 'all')

df.isnull().sum()

df.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace = True)

df = pd.get_dummies(df)

df['Age'] = df['Age'].fillna(df['Age'].mode()[0])

df_x = df.drop('Survived',axis = 1).values
df_y = df['Survived'].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df_x,df_y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=321)
rf.fit(X_train, Y_train)

from sklearn.metrics import f1_score, classification_report
Y_test_pred = rf.predict(X_test)
f1_score(Y_test,Y_test_pred)
#0.7407407407407408

rep = classification_report(Y_test,Y_test_pred)
