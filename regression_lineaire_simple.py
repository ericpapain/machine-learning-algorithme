#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:56:00 2019

@author: eric
"""

""" prediction de salaire en utilisant un algorithme de regression lineaire en macine learning"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importation du jeux de donnée 

data_set_import = pd.read_csv('/home/eric/Documents/Master 2 debut 15 septembre 2019/apprentissages-Le hong phuong/machine learningAZUDEM/datasetfolder/SalaryData_regression.csv')

X = data_set_import.iloc[:,:-1].values
y = data_set_import.iloc[:,-1].values

#(splitting the data set)separer notre jeux de donnée en jeux d'entrainement et en jeu de test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


"""
#pouir la regression lineaire nous ne devons pas normaliser les variables prédictives.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
#ici on utilise la methode de moindre carre pour minimiser l'erreur <som = (y - ŷ)² on minimise au max> on aura un meilleur modèl.

# <Y = b0 + b1*X> dans notre cas on aurra plutot  <salary = bo +b1 * experience> 
# nous devons normaliser ou standartider les valeurs de notre variable dependante et de la variable indépendante
# <y> est un vecteur de variable contrairement a <X> qui est pour sa part une metrique de calcul

"""Fitting <Simple linear regression to the training set>"""
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train,y_train)

#prediction de l'entrainement sur les jeux de données de test
y_prediction = regression.predict(X_test)


#visualisation des resultats de prediction de la regression lineaire et de notre modele 
plt.scatter(X_train,y_train,color='blue')


plt.plot(X_train, regression.predict(X_train), color ='red')

plt.title('salaire vs experience(training set)')

plt.xlabel('anneé d''experience')

plt.ylabel('saliare correspondant')

plt.show()


"""#visualisation des test set resultat
plt.scatter(X_test,y_test,color='blue')

plt.plot(X_train, regression.predict(X_train), color ='red')

plt.title('salaire vs experience(training set) resultat de prediction')

plt.xlabel('anneé d''experience resultt de prediction')

plt.ylabel('saliare correspondant resultat de prediction')

plt.show()"""


