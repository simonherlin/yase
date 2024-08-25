# Guide d'Automatisation des Tâches

## Tâches

De nombreuses tâches, en particulier celles liées à la configuration et aux tests, sont automatisées en utilisant `make`. Ces tâches sont définies dans le `Makefile` à la racine du projet. Vous trouverez ci-dessous les principales tâches disponibles.

L'environnement virtuel par défaut est appelé `venv`. Cet environnement est activé en tant que partie intégrante de ces tâches.

| Tâche                  | Commande                | Description                                                                                     |
|------------------------|-------------------------|-------------------------------------------------------------------------------------------------|
| Installer les dépendances | `make install`           | Crée un environnement virtuel et installe les dépendances définies dans `requirements.txt`.      |
| Exécuter les tests unitaires avec couverture | `make tests`            | Exécute les tests unitaires avec `pytest` et génère un rapport de couverture. Les résultats sont enregistrés dans le dossier `results/`. |
| Formater le code        | `make format`            | Formate le code Python en utilisant `black`.                                                    |
| Vérifier le code avec Flake8 | `make lint`              | Exécute `flake8` pour vérifier le respect des conventions de codage. Les résultats sont enregistrés dans `results/flake8-report.txt`. |
| Lancer l'application principale | `make run`              | Lance l'application principale définie dans `src/yase/main.py`.        |
| Lancer JupyterLab       | `make jupyterlab`       | Démarre JupyterLab pour l'exploration des données et le développement interactif.               |
| Arrêter JupyterLab      | `make stop-jupyterlab`  | Arrête JupyterLab s'il est en cours d'exécution.                                                |
| Analyser le code avec SonarQube | `make sonar`            | Exécute une analyse de code avec SonarQube (si configuré).                                       |
| Nettoyer l'environnement | `make clean`            | Supprime l'environnement virtuel et les fichiers temporaires générés.                           |

### Détails des Commandes Make

1. **Installer les dépendances (`make install`)** :
   - Crée un environnement virtuel dans le dossier `venv` et installe toutes les dépendances définies dans `requirements.txt`.

2. **Exécuter les tests unitaires avec couverture (`make tests`)** :
   - Exécute les tests définis dans le répertoire `tests/` en utilisant `pytest`.
   - Génère un rapport de couverture de code au format HTML, ainsi qu'un fichier JUnit XML pour les résultats des tests.
   - Tous les résultats sont stockés dans le dossier `results/`.

3. **Formater le code (`make format`)** :
   - Utilise `black` pour formater automatiquement le code Python selon les standards définis.

4. **Vérifier le code avec Flake8 (`make lint`)** :
   - Exécute `flake8` pour vérifier la conformité du code aux normes de style Python.
   - Les résultats de `flake8` sont enregistrés dans `results/flake8-report.txt`.

5. **Lancer l'application principale (`make run`)** :
   - Lance le script principal de votre projet. C'est utile pour exécuter rapidement l'application en cours de développement.

6. **Lancer JupyterLab (`make jupyterlab`)** :
   - Démarre un serveur JupyterLab, qui peut être utilisé pour le développement interactif et l'exploration des données.

7. **Arrêter JupyterLab (`make stop-jupyterlab`)** :
   - Arrête le serveur JupyterLab s'il est en cours d'exécution.

8. **Analyser le code avec SonarQube (`make sonar`)** :
   - Exécute une analyse de code avec SonarQube pour évaluer la qualité du code et détecter les vulnérabilités potentielles.
   - Cette commande n'est active que si SonarQube est configuré.

9. **Nettoyer l'environnement (`make clean`)** :
   - Supprime l'environnement virtuel, les fichiers temporaires générés lors des builds, et les caches pour assurer un environnement propre.

### Recommandations

- **Configuration initiale** : Exécutez `make install` dès la création du projet pour configurer l'environnement de développement.
- **Tests réguliers** : Utilisez `make tests` pour vérifier que votre code fonctionne comme prévu et que la couverture des tests est adéquate.
- **Maintenance du code** : Avant chaque commit, il est conseillé d'exécuter `make lint` et `make format` pour garantir la qualité et la conformité du code.

Ce guide vous aidera à automatiser les tâches courantes et à maintenir un environnement de développement efficace et cohérent pour votre projet yase.
