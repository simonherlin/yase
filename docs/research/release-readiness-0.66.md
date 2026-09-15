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
- Sdist complétée avec le code C++, le builder, les tests, exemples et docs.
- Extras alignés : `onnx` CPU et `onnx-gpu` sont des alternatives ; `all` est
  la pile complète CPU ; `all-gpu` ajoute ONNX GPU et TensorRT.
- Version package synchronisée à `0.66.0`.

## Vérifications effectuées

| Vérification | Résultat |
|---|---|
| Ruff | pass |
| Tests unitaires | 147 pass |
| Couverture | 82.44 %, seuil 80 % |
| Wheel portable | `yase-0.66.0-py3-none-any.whl`, sans `_native` |
| Sdist | contient `native/yase_native.cpp` et `tools/build_package.py` |
| Wheel natif | `cp313-cp313-linux_x86_64`, `_native` inclus |
| Installation wheel natif | import, IoU et NMS pass |
| Lockfile | `uv lock --check` pass |

Le poste courant ne possède pas les headers de développement Python 3.12 ni
`python3.12-venv`, donc le chemin local `uv build` isolé ne peut pas être
reproduit tel quel sans installer les paquets système correspondants. Le
profil natif a néanmoins été compilé et installé avec Python 3.13 géré par uv,
dont les headers sont présents ; le problème restant est environnemental et
est couvert par la CI Linux.

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

1. Les wheels natifs multi-OS ne sont pas encore publiés ; la CI doit être
   branchée à cibuildwheel avant une publication PyPI avec accélération.
2. Les performances C++ doivent être benchmarkées contre les fallbacks sur des
   lots représentatifs ; la présence du binaire ne garantit pas un gain pour
   de très petites listes.
3. Les providers CUDA/TensorRT/OpenVINO GPU restent hardware-gated et ne sont
   pas simulés par les tests CPU.
