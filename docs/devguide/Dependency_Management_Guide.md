# Guide de Gestion des Dépendances

La gestion des dépendances du projet est un élément clé pour en faciliter le développement, la livraison et la réutilisation. Plusieurs fichiers dans ce dépôt spécifient les dépendances, chacun ayant un objectif différent.

## Résumé

- **requirements.txt** : Spécification des dépendances nécessaires au développement.
- **pyproject.toml** : Dépendances du code du projet.
  - **[tool.poetry.dependencies]** : Dépendances minimales pour exécuter le code du projet.
  - **[tool.poetry.dev-dependencies]** : Dépendances supplémentaires pour les tests et le développement.

Le code du projet est défini comme le contenu du répertoire `src`.

### Utilisation Simplifiée

- **Développement** : Vous pouvez exécuter toutes les activités de développement après avoir installé les dépendances avec `pip install -r requirements.txt`.
- **Exécution du code** : Exécutez la fonctionnalité principale du projet après avoir installé le package avec `pip install dist/<nom_du_package>.whl`.
- **Tests et Analyse Statique** : Exécutez tous les tests unitaires et l'analyse statique après avoir installé les dépendances avec `pip install .[dev]`.

### Code du Projet

Les dépendances du code du projet sont définies dans le fichier `pyproject.toml`, en accord avec les meilleures pratiques actuelles. Elles incluent les dépendances minimales requises pour exécuter le code du projet.

- **[tool.poetry.dependencies]** : Dépendances minimales nécessaires pour exécuter le code du projet (dans le répertoire `src`).
- **[tool.poetry.dev-dependencies]** : Dépendances supplémentaires pour les tests unitaires, l'analyse statique, etc.

### Environnement de Développement

L'environnement de développement est décrit dans le fichier `requirements.txt`, qui couvre toutes les dépendances (pas seulement les bibliothèques Python) nécessaires pour les activités des développeurs sur le projet.
