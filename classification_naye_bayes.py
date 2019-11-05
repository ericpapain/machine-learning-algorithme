#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:04:03 2019

@author: eric
"""

"""
utilisation de la methode de naive bayes pour la classification de mail et de spam
qui est l'une des meilleures algorithme de machine learning utilisé pour la classification 
et le traitement des texte
"""

import os
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# quelques variables globales utiles
PATH_TO_HAM_DIR = '/home/eric/Documents/Master 2 debut 15 septembre 2019/apprentissages-Le hong phuong/tp de classe/naive-bayes-spam-classifier-machine-learning-master/emails/ham'
PATH_TO_SPAM_DIR = '/home/eric/Documents/Master 2 debut 15 septembre 2019/apprentissages-Le hong phuong/tp de classe/naive-bayes-spam-classifier-machine-learning-master/emails/spam'

SPAM_TYPE = "C'est un SPAM!!"
HAM_TYPE = "C'est un bon mail!!"

#les tableaux X et Y seront de la meme taille et ordonnes
X = [] # represente l'input Data (ici les mails)
#indique s'il s'agit d'un mail ou non
Y = [] #les etiquettes (labels) pour le training set


def readFilesFromDirectory(path, classification):
    os.chdir(path)
    files_name = os.listdir(path)
    for current_file in files_name:
        message = extract_mail_body(current_file)
        X.append(message)
        Y.append(classification)
        

#fonction de lecture du contenu d'un fichier texte donne.
#ici, on fait un peu de traitement pour ne prendre en compte que le "corps du mail".
# On ignorer les en-tetes des mails
def extract_mail_body(file_name_str):
    inBody = False
    lines = []
    file_descriptor = io.open(file_name_str,'r', encoding='latin1')
    for line in file_descriptor:
        if inBody:
            lines.append(line)
        elif line == '\n':
            inBody = True
        message = '\n'.join(lines)
    file_descriptor.close()
    return message


#appel de la fonction de chargement des mails (charger les mail normaux ensuite les SPAM)
readFilesFromDirectory(PATH_TO_HAM_DIR, HAM_TYPE)

readFilesFromDirectory(PATH_TO_SPAM_DIR, SPAM_TYPE)

training_set = pd.DataFrame({'X': X, 'Y': Y})

#separation du jeux de données en jeux de test et en jeux d'entrainement
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#convertion du texte en une matrix de jeton
vectorizer = CountVectorizer()
"""
counts = vectorizer.fit_transform(training_set['X'].values)
"""
#transformation des données d'entrainement et de test

counts_X_train = vectorizer.fit_transform(X_train)

#lancement des entrainements
classifier = MultinomialNB()

targets = training_set['Y'].values
classifier.fit(counts_X_train, y_train)


#prediction sur les jeux de test
counts_X_test = vectorizer.transform(X_test)

#parcours de chaque mail et prediction de sa classe

"""for each_mail in counts_X_test:
    result = classifier.predict(each_mail)
    tab_result.append(result)
    """
#verification des resultats
#prediction_jeux_de_test = classifier.predict(counts_X_test)

examples = ['votre inscription a été prise en compte', "Hi Bob, how about a game of golf tomorrow?"]

example_counts = vectorizer.transform(X_test)

predictions = classifier.predict(example_counts)

print (predictions)

valeurs_predicte = np.asarray(predictions)
valeurs_attendue = np.asarray(y_test)


"""comparaison entre les valeurs obtenue et les valeurs de prediction attendue par le 
jeux de test"""
tab_verif = []


for i in range(len(predictions)):
    if(valeurs_predicte[i]==valeurs_attendue[i]):
        tab_verif.append(1)
    else:
        tab_verif.append(0)

#compte des faux resultats
bon=0
mauvais=0
for elt in tab_verif:
    if(elt==1):
        bon=bon+1
    else:
        mauvais=mauvais+1

"""calcul de pourcentage de prediction"""

def fonction_Pourcentage(nbre_mail_de_test, nbre_bon):
    pourcentage = (nbre_bon*100)/nbre_mail_de_test
    return pourcentage

print('#############################################################################')

print('nous avons obtenue pres de ',bon,' bon mail sur ', len(predictions), ' mail de test')
print('nous avons obtenue pres de ',mauvais,' mauvaise prediction de mail sur ', len(predictions), ' mail de test')

print('##################################\n'
      '#la precision est donc de', fonction_Pourcentage(len(predictions),bon),'% #\n##################################\n')
