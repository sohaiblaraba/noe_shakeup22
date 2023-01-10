<!-- A PROPOS -->
# A propos

Ceci est une démo pour le projet NOE, dans le cadre du Shake'UP 2022.
L'objectif étant de développer un modèle de classification de déchets pour permettre de mieux recycler.

<!-- ENVIRONNEMENT -->
# Environnement
Avant d'utiliser le projet, il est nécessaire de préparer l'environnement de travail.

## Installer Anaconda
* Pour installer Anaconda, il suffit de le télécharger du [site officiel](https://www.anaconda.com/products/distribution) et l'installer.
* Ensuite, il faut créer un environnement. Pour ce faire, il faut utiliser la commande suivate:
`conda create -n nom_environnement`
(Vous pouvez choisir le nom que vous voulez)
* Activer cet environnement en faisat: `conda activate nom_environnement`

## Dépendences
Une fois l'environnement est créé et est installer, il faut installer les dépendances nécessaires pour faire fonctionner le projet.
* D'abord, il faut s'assurer que la machine contient une carte graphique capable de faire un entrainement.
Il faut installer [CUDA  11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) et [CUDNN 8.2.1](https://developer.nvidia.com/rdp/cudnn-archive)

* Les dépendances sont listées dans le fichier "requirements.txt". Il faut les installer en faisant:
`pip install requirements.txt`

<!-- ENTRAINEMENT -->
# Entrainement du modèle
Pour lancer un entrainement d'un modèle, il faut lancer la commande suivante:\
`python train.py --data path_to_data --split 0.2 --model ResNet50 --batch 32 --epoch 10 --save_model mymodel.h5` 
Où:
* --data path_to_data: Il faut spécifier le chemin vers le dossier contenant les images.
Le dossier doit avoir des sousdossiers (qui correspondent aux classes), qui contiennent les images correspondantes, comme montré ci-dessous :
Vous pouvez déjà télécharger [cette base de données](https://drive.google.com/file/d/1tBfsf7ghNRGjSDx3IXpRw5f6NNoGl-OX/view?usp=sharing) pour commencer.
data-\
------- classe1 \
----------------- image1\
----------------- image2\
..\
----------------- imagen\
------- classe2\
----------------- image1\
----------------- image2\
..\
----------------- imagen\
...\
------- classem\
----------------- image1\
----------------- image2\
..\
----------------- imagen

* --split 0.2 : Le pourcentage de données utilisées en validation. 0.2 => 20% de données.

* --model ResNet50 : Modèles à entrainer. Liste de modèles disponibles: ResNet50, DenseNet121, MobileNet, MobileNet_V2, MobileNetV3Small
* --batch 32 : Taille du batch
* --epoch 10 : Nombre d'epochs
* --save_model mymodel.h5 : nom du modèle à sauvegarder après entrainement.

<!-- PREDICTION -->
# Prédiction sur une image
`python predict.py --model ResNet50 --image image.jpeg` 

<!-- REALTIME -->
# Temps réel
`python realtime.py` 
