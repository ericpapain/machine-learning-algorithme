# Description de chaque projet

## moteur de recherche d'image dans la base de donnée de Wang
    > Enoncé 
La recherche d’images par le contenu est un domaine de recherche très actif depuis plusieurs
années maintenant. Avec l’explosion de l’imagerie numérique, nous nous retrouvons souvent
avec d’énormes bases d’images et la recherche d’une image particulière est un problème difficile à résoudre.
Dans ce projet, Nous :
- implémentons un petit système de recherche d’images par le contenu avec des descripteurs SIFT. L’idée est de calculer des caractéristiques pour chaque image en entrée et de rechercher les images ayant les caractéristiques les plus semblables.
- La base d’images utilisée est constituée de 1000 images séparé en 100 classes et chaque classe ayant 100 éléments et tous classé par concept (Afrique, elephant, ...)
    > Technologies, outils et divers...
- Python
- K-means
- SIFT
- BoVW
- KNN
- Tensorflow
- Keras
- Theano
- sklearn
- dataset utilisé  http://wang.ist.psu.edu/docs/related/
- résultats ( sont disponible dans le rapport en pdf dans le dossier du projet)
