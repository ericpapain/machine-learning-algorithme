#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:26:47 2019

@author: eric
"""

"""
dans se jeux de donnée on va implementer la regression lineaire multiple pour predire le profit dans une 
startup ici nous avons plusieurs variable c'est sa l'avantage"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importation du jeux de donnée 

data_set_import = pd.read_csv('/home/eric/Documents/Master 2 debut 15 septembre 2019/apprentissages-Le hong phuong/machine learningAZUDEM/datasetfolder/Startups_multilinear_regression.csv')

#independant variable
X = data_set_import.iloc[:,:-1].values

#dependant variable
y = data_set_import.iloc[:,-1].values


#(encoding categorial data)traitement des variable cathégoriel qui son du texte nous devons tranformer tous en meme valeurs 
#la variable dependante est la variable a predire et la variable independante celle qui permet de predire

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encodeur_X = LabelEncoder()
X[:,3]=label_encodeur_X.fit_transform(X[:,3])

#(Dummy variable ) ou variable catégorielle :on va creer les dummy encoding pour eviter que notre jeux de donner donne plsu d'importance au variable on utilise le one hot encodeur
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap
"""pour que nous ayons moins de variable dans le tableau apres convertion on dois passer les donnée
de facon manuelle pour provoquer ...."""

"""X = X[:,1:]"""


#(splitting the data set)separer notre jeux de donnée en jeux d'entrainement et en jeu de test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""ici nous n'avons pas besoin de passez les donnée a la normalisation ou standardisation car la librairie le feras a notre place"""

#fitting multiple Linear regression to the training set
from sklearn.linear_model import LinearRegression

regressionMultiple = LinearRegression()

regressionMultiple.fit(X_train, y_train)


#prediction des test apres entrainement
y_prediction = regressionMultiple.predict(X_test)


########################################################
# Building the optimal model using backward elimination#
########################################################
""" ici on essaye de selectionner les varible independante les plus predictive
se qu'on appelle le backward elimination en modifiant la valeurs et la puissance d'importance des entrée
"""

import statsmodels.api as sm


""" ici on ajoute une premiere ligne dans notre jeux d'entrainement pour 
essayer de faire modifier les valeurs des variables et de voir les meilleurs predicteurs de la 
variable dependante qui est le profil"""

#np.ones((50, 1)) cree 50 ligne et 1 colone
#astype(int) convertion de type 
#la premiere ligne ajouter correspond a la valeur X0=1
"""cette methode peremet de supprimer les variable independante one by one pour avaoir les
variable les plus significative pour la prediction """
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#selection des variables optimale
""" le principe consiste a ajouter toutes les variable independante dans la liste et de supprimer
un par un ceux ou celle qui sont les moins significante <Ordinary Least Squares>"""

"""pour selectionner la variable on dois choisir celle donc <significative level >0.05>> on vois cella dans le tableau avec summary"""
""" apres chaque itération on retire l'index de celui ayant la plus 
grande probabilité"""
X_optimal = X[:,[0,1,2,3,4,5]]
regressionMultiple_OLS = sm.OLS(endog =y, exog=X_optimal).fit()
regressionMultiple_OLS.summary()

#supresion de la variable x4 car il a une forte probabilité
X_optimal = X[:,[0,1,3,5]]
regressionMultiple_OLS = sm.OLS(endog =y, exog=X_optimal).fit()
regressionMultiple_OLS.summary()

#supresion de la variable x1 car il a une forte probabilité
X_optimal = X[:,[0,3,5]]
regressionMultiple_OLS = sm.OLS(endog =y, exog=X_optimal).fit()
regressionMultiple_OLS.summary()

#supresion de la variable x5 car il a une forte probabilité
X_optimal = X[:,[0,3]]
regressionMultiple_OLS = sm.OLS(endog =y, exog=X_optimal).fit()
regressionMultiple_OLS.summary()

###############################
#conclusion: la varible restante x3 est le meilleur predicteur pour le modele construit
##################################################################################

""" a chaque temps il faut changer les valeur et voir la variation 
de la variance et de l'ecart type dans le tableau."""