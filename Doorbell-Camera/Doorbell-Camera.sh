#! /bin/sh


# Allez dans le répertoire de l'application
cd /home/demoia/DemoIA/Doorbell-Camera

# Configuration de la caméra à la première utilisation
./camera_setup.sh

# Lancement de l'application
# Appuyer sur "Echap" pour quitter
./doorbell_camera.py

