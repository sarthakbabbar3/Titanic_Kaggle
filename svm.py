# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:53:44 2018

@author: SARTHAK BABBAR
"""


import pandas as pd 
import numpy as np
from sklearn import preprocessing,svm


df= pd.read_csv("train.csv")
#df= pd.read_csv("smotex.csv")

df=df[['Sex','Survived','Pclass','Age',	'SibSp','Parch','Fare','Embarked']]

df.replace(r'\s+', np.nan, regex=True)

df=df.fillna(df.median())

X_train=np.array(df.drop(['Survived'],1))

X_train=preprocessing.scale(X_train)

y_train =np.array(df['Survived'])

clf=svm.SVC()
clf.fit(X_train,y_train)

pf= pd.read_csv("test.csv")

pf=pf[['Sex','Pclass','Age',	'SibSp','Parch','Fare','Embarked']]

pf.replace(r'\s+', np.nan, regex=True)

pf=pf.fillna(pf.mean())

X_test=np.array(pf)

X_test=preprocessing.scale(X_test)

y_pred = clf.predict(X_test)
np.savetxt('Predictions.csv',y_pred, delimiter=',')