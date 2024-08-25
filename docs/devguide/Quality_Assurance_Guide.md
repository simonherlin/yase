
# Guide d'Assurance Qualité

Ce guide décrit les pratiques d'assurance qualité mises en place pour **yase**. L'objectif est de garantir que le code est maintenu à un haut niveau de qualité tout au long du cycle de vie du projet.

## Linting

Le linting est un processus qui analyse le code pour détecter des erreurs de style, des erreurs de programmation simples, ou des problèmes d'intégration potentiels.

- **Flake8** est utilisé pour effectuer le linting du code Python :
  ```bash
  make lint
  ```

  - **Sortie** : Les résultats du linting sont enregistrés dans un fichier texte `results/flake8-report.txt`. Ce fichier contient des détails sur les violations des conventions de style et des erreurs potentielles dans le code.

## Tests Unitaires

Les tests unitaires permettent de vérifier que chaque composant individuel du code fonctionne comme prévu.

- **Pytest** est utilisé pour exécuter les tests unitaires :
  ```bash
  make tests
  ```

  - **Sortie** : Les résultats des tests sont enregistrés dans `results/pytest-results.xml` au format JUnit XML, et un rapport de couverture de code est généré en HTML.

  - **Rapport de Couverture** : La couverture de code est un indicateur du pourcentage de votre code qui est couvert par les tests. Ce rapport est généré en HTML et se trouve dans le dossier `htmlcov`.

## Formatage du Code

Le formatage du code assure que tout le code source est conforme à un style cohérent, ce qui facilite la lecture et la maintenance.

- **Black** est utilisé pour formater le code Python automatiquement :
  ```bash
  make format
  ```

  - **Sortie** : Le code sera automatiquement réorganisé selon les conventions de style définies par Black. Cela inclut la longueur maximale des lignes, l'utilisation des guillemets, etc.

## Analyse Statique

L'analyse statique est utilisée pour examiner le code sans l'exécuter, afin de détecter des erreurs potentielles, des problèmes de sécurité, ou des mauvaises pratiques de programmation.

- **SonarQube** (si configuré) est utilisé pour effectuer une analyse statique approfondie du code. SonarQube permet également de suivre les mesures de qualité du code comme la couverture des tests, la duplication du code, et les vulnérabilités potentielles :
  ```bash
  make sonar
  ```

  - **Sortie** : Les résultats de l'analyse statique seront disponibles dans l'interface SonarQube configurée, où vous pouvez explorer les problèmes trouvés et suivre les progrès dans la qualité du code au fil du temps.

## Bonnes Pratiques

- **Commit régulier** : Assurez-vous de committer fréquemment vos modifications, en particulier après avoir passé les tests et l'analyse statique. Cela permet de s'assurer que le code est toujours dans un état stable.

- **Revue de Code** : Avant d'intégrer les changements majeurs dans la branche principale, effectuez des revues de code avec votre équipe pour détecter d'éventuels problèmes ou pour améliorer la qualité du code.

- **Tests exhaustifs** : Assurez-vous que chaque nouvelle fonctionnalité ou correction de bug est couverte par des tests unitaires. Cela garantit que les modifications futures n'introduisent pas de régressions.

## Conclusion

L'assurance qualité est essentielle pour maintenir un projet robuste et durable. En utilisant les outils et les processus décrits dans ce guide, vous pouvez vous assurer que votre code reste de haute qualité tout au long de son développement.

Si vous avez des questions ou des suggestions concernant ce guide, n'hésitez pas à ouvrir une issue ou à contribuer directement au projet.