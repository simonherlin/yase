# Yase 0.66.0 — audit architecture et distribution

Date : 2026-09-15

## Décision d’architecture

Le C++ local reste volontairement limité aux kernels CPU indépendants du
framework : matrice IoU et NMS glouton. Ce sont des opérations de post-
traitement et de tracking réutilisées par plusieurs backends. L’inférence
elle-même doit rester dans le runtime spécialisé choisi par l’application
(ONNX Runtime, OpenVINO, TensorRT ou PyTorch). Réécrire ces moteurs dans Yase
rendrait le package plus lourd, moins portable et moins fiable sans améliorer
la qualité des modèles.

Le contrat public est donc :

- cœur Python portable, sans framework d’inférence obligatoire ;
- accélérateur C++ optionnel avec fallback Python déterministe ;
- wheel pur par défaut, installable sans compilateur ;
- wheel platform/ABI spécifique quand `YASE_BUILD_NATIVE=1` est défini ;
- construction multi-plateforme native à confier à cibuildwheel.

## Changements 0.66.0

- Migration du backend de build vers setuptools, nécessaire pour déclarer
  proprement l’extension C++ optionnelle dans PEP 517.
- Ajout de `setup.py` limité à la configuration programmatique de cette
  extension ; les métadonnées restent dans `pyproject.toml`.
- Ajout de `tools/build_package.py`, `make build` et `make build-native`.
- Ajout de `tools/build_wheels.py` et de la matrice cibuildwheel externe
  Linux/macOS/Windows, sans pipeline hébergée.
- Ajout de la validation stricte ONNX (`strict_providers=True`) et de la
  configuration/cache compilée OpenVINO.
- Extension de `runtime_smoke.py` aux devices OpenVINO sélectionnés et aux
  providers ONNX explicitement demandés.
- Sdist complétée avec le code C++, le builder, les tests, exemples et docs.
- Extras alignés : `onnx` CPU et `onnx-gpu` sont des alternatives ; `all` est
  la pile complète CPU ; `all-gpu` ajoute ONNX GPU et TensorRT.
- Version package synchronisée à `0.66.0`.

## Vérifications effectuées

| Vérification | Résultat |
|---|---|
| Ruff | pass |
| Tests unitaires | 159 pass |
| Couverture | 82.78 %, seuil 80 % |
| Wheel portable | `yase-0.66.0-py3-none-any.whl`, sans `_native` |
| Sdist | contient `native/yase_native.cpp` et `tools/build_package.py` |
| Wheel natif | `cp312-cp312-linux_x86_64`, `_native` inclus |
| Installation wheel natif | import, IoU et NMS pass dans un venv isolé |
| Wheel portable Python | installation/import validés sur Python 3.12 ; matrice 3.10/3.11/3.13 à rejouer dans release externe |
| Smoke OpenVINO | CPU + GPU réels sur le modèle synthétique local |
| Smoke ONNX | CPU réel ; CUDA non installé sur le poste |
| Micro-benchmark C++ | 64 boîtes : IoU ~28×, NMS ~33× contre fallback local |
| Lockfile | `uv lock --check` pass |

Le poste courant ne possède pas `python3.12-venv` ni les headers système par
défaut : le chemin `make build-isolated` nécessite donc les paquets système
correspondants. `make build` réutilise volontairement l’environnement `uv` et
fonctionne localement. Les headers Ubuntu ont été extraits dans un répertoire
utilisateur pour compiler et installer la wheel native cp312 ; aucun paquet
système n’a été ajouté avec privilèges root.

## Parcours utilisateur recommandé

```bash
# installation légère et universelle
uv pip install yase

# depuis les sources, wheel pur
make build

# depuis les sources, wheel optimisé pour l’hôte
make build-native

# matrice de publication multi-OS et multi-Python
uvx cibuildwheel
```

Les poids de modèles ne sont jamais téléchargés implicitement. Les contraintes
CUDA, TensorRT, licences de checkpoints et compatibilités de provider restent
des choix de déploiement visibles dans `yase diagnostics --providers`.

## Risques résiduels

1. Les wheels natifs multi-OS ne sont pas encore publiés ; un environnement de
   release dédié doit lancer cibuildwheel avant une publication PyPI avec
   accélération. GitHub Actions reste volontairement désactivé et la release
   doit être exécutée depuis un poste/runner externe contrôlé.
2. Le micro-benchmark C++/fallback est validé sur les kernels ; il reste à
   mesurer l’impact end-to-end avec des lots, résolutions et modèles réels.
   La présence du binaire ne garantit pas un gain pour de très petites listes.
3. Les providers CUDA/TensorRT/OpenVINO GPU restent hardware-gated et ne sont
   pas simulés par les tests CPU.
