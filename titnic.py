#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:54:16 2020

@author: hms
"""


import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("./train.csv")
train_data.head()
test_data = pd.read_csv("./test.csv")
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

Data = [train_data, test_data]
for dataset in Data:
    dataset['Title'] = dataset['Name'].str.extract('([A-za-z]+)\.', expand=False)
    
title_mapping = {"Mr":0, "Miss":1, "Mrs":2,
                "Master":3, "Dr":3, "Rev":3, "Col": 3, 'Ms': 3, 'Mlle': 3, "Major": 3, 'Lady': 3, 'Capt': 3,
                 'Sir': 3, 'Don': 3, 'Mme':3, 'Jonkheer': 3, 'Countess': 3 }
for dataset in Data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
# 데이터 셋에서 불필요한 feature 삭제
train_data. drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)

sex_mapping = {"male": 0, "female":1}
for dataset in Data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    
# Missing Age를 각 Title에 대한 연령의 중간값 으로 채운다(Mr, Mrs, Miss, Others)
train_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('median'), inplace=True)
test_data['Age'].fillna(test_data.groupby('Title')['Age'].transform('median'), inplace=True)



Pclass1 = train_data[train_data['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

for dataset in Data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in Data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    
train_data["Fare"].fillna(train_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test_data["Fare"].fillna(test_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)


for dataset in Data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    
Pclass1 = train_data[train_data['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))


cabin_mapping = {'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F':2, 'G':2.4, 'T': 2.8}
for dataset in Data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    
    
train_data['Cabin'].fillna(train_data.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test_data['Cabin'].fillna(test_data.groupby('Pclass')['Cabin'].transform('median'), inplace=True)


features_drop = ['PassengerId', 'Ticket']
train = train_data.drop(features_drop, axis=1)
test = test_data.drop(features_drop, axis=1)


y = train['Survived']
X = train.drop('Survived', axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True, random_state=1004)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_sc, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train_sc, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test_sc, y_test)))