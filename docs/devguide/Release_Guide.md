# Guide de Publication

Ce guide décrit le processus de publication pour ce projet.

## Processus de Publication

1. **Vérifications avant publication** :
    - Assurez-vous que le numéro de version dans `pyproject.toml` est à jour.
    - Vérifiez que toutes les pull requests requises sont fusionnées.
    - Assurez-vous que vous êtes sur la branche principale (`main`) et que votre copie locale est à jour avec le dépôt Git.
    - Vérifiez qu'il n'y a aucun fichier non commis dans votre copie de travail.

2. **Ajouter un tag Git** :
    - Ajoutez un tag Git de la forme `<version>` et poussez-le vers le dépôt.

3. **Créer les fichiers de publication** :
    - Exécutez `make build` pour créer le package.

4. **Vérifier les fichiers de publication** :
    - Assurez-vous que les fichiers de publication sont complets et fonctionnels.

5. **Distribuer le package** :
    - Envoyez les fichiers de publication au client ou téléchargez-les sur le dépôt de packages.

## Package de Release

Une fois qu'un utilisateur dispose d'un environnement fonctionnel, le code du projet peut être mis à jour hors ligne en utilisant un package `wheel`.

1. **Créer le package `wheel`** :
    - Exécutez `make build`. Le fichier `wheel` sera créé dans le répertoire `dist/`.

2. **Installation du package** :
    - Pour installer le package, exécutez la commande suivante dans l'environnement virtuel :
      ```bash
      pip install dist/<nom_du_package>.whl
      ```

