#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 05:34:25 2019

@author: eric
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importation du jeux de donnée

data_set_import = pd.read_csv('/home/eric/Documents/Master 2 debut 15 septembre 2019/apprentissages-Le hong phuong/machine learningAZUDEM/datasetfolder/data.csv')

X = data_set_import.iloc[:,:-1].values
y = data_set_import.iloc[:,-1].values


#(missing values)traitement de valeurs manquantes par la moyenne des elements de la colone avec sklearn

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3]) 
X[:,1:3] = imputer.transform(X[:,1:3])

"""
implementation de l'algorithme de somme
som = 0
for i in range (len(data_set_import)):
    if (X[i,2] != 'nan'):
       som=som+X[i,2]
       
moye = som/(len(data_set_import))
som/10"""


#(encoding categorial data)traitement des variable cathégoriel qui son du texte nous devons tranformer tous en meme valeurs 
#la variable dependante est la variable a predire et la variable independante celle qui permet de predire

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encodeur_X = LabelEncoder()
X[:,0]=label_encodeur_X.fit_transform(X[:,0])

label_encodeur_y = LabelEncoder()

#(Dummy variable ) ou variable catégorielle :on va creer les dummy encoding pour eviter que notre jeux de donner donne plsu d'importance au variable on utilise le one hot encodeur
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
y = label_encodeur_y.fit_transform(y)


#(splitting the data set)separer notre jeux de donnée en jeux d'entrainement et en jeu de test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#(scaling variable) equilibrage des valeurs pour eviter d'avoir des variables tres expressive et plus importante que d'autre
"""L'écart-type vous indique l'étendue des données. C'est une mesure de la distance qui sépare
chaque valeur observée de la moyenne. Quelle que soit la distribution, 
environ 95% des valeurs se situeront à moins de 2 écarts-types de la moyenne."""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)