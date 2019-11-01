#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:46:55 2019

@author: eric
"""
"""
dans cette section comme nous n'avons pas beaucoup de donnée nous n'allons 
pas diviser le jeux de données en plusieurs jeux de test et d'entrainement
nous allons travailler avec les données directement comme donnée d'entrainement 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importation du jeux de donnée

data_set_import = pd.read_csv('/home/eric/Documents/Master 2 debut 15 septembre 2019/apprentissages-Le hong phuong/machine learningAZUDEM/datasetfolder/Position_Salaries_polynomial_regression.csv')

X = data_set_import.iloc[:,1:2].values
y = data_set_import.iloc[:,2].values

"""
ici egalement nous pouvons constater que la varible des postes correspond a leurs id
en termes de payement alors dans se cas nous allons prendre la variable id comme predicteur 
et voir la distribution comparer la regression lineaire simple avec celle polynomiale
qui matche bien les points.
"""

# utilisation du modele de regression lineaire pour essayer d'aficher
from sklearn.linear_model import LinearRegression
regression_lineaire = LinearRegression()
regression_lineaire.fit(X, y)


#utilisation du modele polynomial qui est dans le package preprocessing plutot
"""
se modele permet de prendre le degre de la fonction et de l'approximer jusqua se qu'elle a l'allure 
de nos données
"""
from sklearn.preprocessing import PolynomialFeatures

regression_polynomial = PolynomialFeatures(degree=4)

X_optimal_data = regression_polynomial.fit_transform(X)

#entrainement sur des valeurs de X_optimaux
regression_polynomial.fit(X_optimal_data, y)

regression_lineaire_X_optimal = LinearRegression()
regression_lineaire_X_optimal.fit(X_optimal_data, y)

#visualisation regresion lineaire 
plt.scatter(X, y, color='red')
plt.plot(X, regression_lineaire.predict(X), color='blue')
plt.title('affichage avec la regression lineaire')
plt.xlabel('ranking de la personnalité')
plt.ylabel('salaire moyen annuelle')

plt.show()

#visualisation regression lineaire optimal
"""plt.scatter(X_optimal_data, y[:,np.newaxis], color='red')
plt.plot(X_optimal_data, regression_lineaire_X_optimal.predict(X_optimal_data))
plt.title('affichage avec la regression lineaire simple')
plt.xlabel('ranking de la personnalité')
plt.ylabel('salaire moyen annuelle')

plt.show()"""

#visualisation regression lineaire optimal
plt.scatter(X, y, color='red')
plt.plot(X, regression_lineaire_X_optimal.predict(X_optimal_data), color='blue')
plt.title('affichage avec la regression polynomial')
plt.xlabel('ranking de la personnalité')
plt.ylabel('salaire moyen annuelle')

plt.show()


#prediction d'une nouvelle valeur
val =[]
val.append(11)
val = np.asarray(val)
val = val[:,np.newaxis]
regression_lineaire.predict(val)
valeur = regression_lineaire_X_optimal.predict(regression_polynomial.fit_transform(val))
valeur = 