# Yase — analyse complète du reste à faire

Date : 2026-09-15 · base : `0.66.0`

Ce document sépare les problèmes bloquants, les améliorations de production
et les extensions de périmètre. Le but est d’éviter de transformer Yase en
un assemblage de modèles sans contrats stables.

## État actuel

| Domaine | État | Preuve / limite |
|---|---|---|
| API image | prêt | `Yase.extract`, chemins, PIL, NumPy, limites |
| Lots d’images | prêt | ordre conservé, batch natif, fallback parallèle |
| Vidéo fichier | prêt | stride, FPS, batch, tracking, sinks, checkpoints |
| Temps réel | prêt pour intégration | worker latest-frame/backpressure, matériel à mesurer |
| Schéma sémantique | prêt pour 1.x | `SemanticResult` et `ObservationBundle` versionnés |
| Backends | extensible | ONNX, OpenVINO, TensorRT, TorchScript, Transformers, OCR |
| Modèles | explicites | aucun poids téléchargé implicitement |
| C++ | ciblé | IoU/NMS, fallback Python, wheel ABI optionnel |
| Packaging | prêt localement | wheel pur 3.10–3.13, wheel natif Linux 3.13 validé |
| Service ASGI | prêt comme boundary | auth/rate-limit laissés au reverse-proxy |
| Observabilité | solide localement | métriques et spans, corrélation distribuée encore à intégrer |
| CI hébergée | désactivée | choix volontaire pour ne pas consommer de crédits |

## Décision Python/C++

Il ne faut pas convertir tout le projet en C++. Les parties qui doivent rester
en Python sont l’orchestration, les contrats, la conversion d’entrées, les
plugins, la configuration, la sérialisation, la vidéo et les politiques
d’erreur. Elles sont dominées par les I/O, les appels aux runtimes et la
flexibilité d’intégration.

Le C++ est pertinent pour les boucles CPU sans dépendance de framework,
réutilisées à haute fréquence : IoU, NMS et éventuellement certains kernels de
tracking après mesure. L’inférence lourde doit rester dans les moteurs
spécialisés qui disposent déjà de kernels CUDA/CPU/NPU et de graph optimizers.
Ajouter du C++ sans benchmark risquerait d’augmenter la surface ABI sans gain
réel sur les petites listes.

## Tâches P0 — nécessaires avant une publication générale

1. Produire les wheels natifs Linux/macOS/Windows et Python 3.10–3.13 dans un
   environnement de release dédié via cibuildwheel. Le dépôt contient déjà la
   configuration ; GitHub Actions ne doit pas être réactivé pour cela.
2. [partiellement livré] Ajouter un benchmark reproductible séparant
   preprocessing, inférence, post-processing et latence bout-en-bout pour
   ONNX/OpenVINO/TensorRT/TorchScript. `record_timings=True` et
   `BenchmarkReport.phase_timings` couvrent désormais les phases de
   l’adaptateur ; la clé `phase_timings` des rapports couvre ces phases, tandis
   que les transferts device explicites et les mesures GPU natives
   restent à valider sur les machines cibles.
3. Valider les providers GPU sur des machines compatibles : CUDA EP, TensorRT,
   OpenVINO GPU/NPU. Le poste local Maxwell ne peut pas être la preuve d’un
   TensorRT moderne.
4. Publier une matrice de compatibilité par OS, Python, runtime, architecture
   CPU/GPU et licence de checkpoint.
5. Ajouter une procédure de migration pour les versions futures de résultats,
   bundles et checkpoints, avec fixtures de versions précédentes.

## Tâches P1 — production et exploitation

1. Ajouter une stratégie de cache explicite pour modèles et sessions, avec
   cycle de vie, fermeture et limites mémoire ; ne jamais télécharger sans
   action explicite de l’application.
2. Propager les contextes de trace dans les sinks, batches et appels de
   backends afin de relier une frame à ses stages.
3. Ajouter timeouts, quotas et annulation au boundary ASGI ; l’authentification
   et le rate-limit global restent idéalement au proxy ou à la plateforme.
4. Enrichir les checkpoints avec une politique de migration et une empreinte
   du modèle/configuration qui a produit l’état.
5. Ajouter une file multi-étages optionnelle pour les flux temps réel lorsque
   le modèle, l’OCR et le VLM ont des cadences différentes.
6. Ajouter un index local HNSW/FAISS optionnel derrière le contrat actuel,
   sans mélanger les espaces d’embedding.

## Tâches P2 — extensions de produit

1. Document AI : layout, tableaux, formulaires et PDF multi-pages.
2. VLM vidéo keyframe-aware avec budget de tokens et mémoire temporelle.
3. Connecteurs Kafka/Redis/objet en plugins séparés.
4. Audio/ASR et synchronisation audio-vidéo si le périmètre devient réellement
   multimodal.
5. Packs de modèles versionnés avec licences et benchmarks publiés.

Le benchmark `benchmarks/native.py` couvre désormais la comparaison native /
fallback des kernels locaux. Il reste à instrumenter les runtimes complets,
car leur latence dépend du modèle, de la résolution, du provider et du
matériel.

## Ce qui ne doit pas être fait maintenant

- Ajouter tous les modèles disponibles sans contrat de sortie ou politique de
  licence.
- Rendre PyTorch, CUDA, OpenCV ou TensorRT obligatoires dans le cœur.
- Mettre les poids dans le wheel Python.
- Réactiver une pipeline hébergée tant que les crédits ne sont pas disponibles.
- Promettre des performances GPU sans mesure sur le matériel correspondant.
- Remplacer le fallback Python par un binaire obligatoire.

## Commande de validation locale

```bash
uv sync --dev
make check
make runtime-smoke       # si ONNX Runtime/OpenVINO sont installés
make build
make build-native        # headers Python + compilateur C++17 requis
```

Le prochain lot recommandé est le benchmark reproductible, puis la procédure
de release cibuildwheel hors GitHub Actions. Les tâches P1 ne doivent être
implémentées qu’après avoir gardé les contrats image/batch/vidéo et la
compatibilité du wheel portable.
