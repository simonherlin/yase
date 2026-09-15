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
| Backends | extensible | ONNX, OpenVINO, TensorRT, TorchScript, Transformers, OCR; providers vérifiables |
| Adaptation matérielle | livré | `HardwareProfile`, sélection `model="auto"`, fallback observable; matrice multi-OS à exécuter sur les hôtes cibles |
| Modèles | explicites | aucun poids téléchargé implicitement |
| C++ | ciblé | IoU/NMS, fallback Python, wheel ABI optionnel |
| Packaging | prêt localement | wheel pur 3.10–3.13, wheel natif Linux 3.13 validé, release externe outillée |
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

1. [outillage livré, validation externe requise] Produire les wheels natifs
   Linux/macOS/Windows et Python 3.10–3.13 dans un environnement de release
   dédié via cibuildwheel. `make wheels`, `tools/build_wheels.py` et le profil
   explicite `cp310`–`cp313` sont maintenant fournis ; chaque OS/toolchain doit
   encore exécuter la matrice sur sa machine correspondante.
2. [partiellement livré] Ajouter un benchmark reproductible séparant
   preprocessing, inférence, post-processing et latence bout-en-bout pour
   ONNX/OpenVINO/TensorRT/TorchScript. `record_timings=True` et
   `BenchmarkReport.phase_timings` couvrent désormais les phases de
   l’adaptateur ; la clé `phase_timings` des rapports couvre ces phases, tandis
   que les transferts device explicites et les mesures GPU natives
   restent à valider sur les machines cibles.
3. [partiellement livré] Valider les providers GPU sur des machines
   compatibles : CUDA EP, TensorRT, OpenVINO GPU/NPU. Le poste local a validé
   OpenVINO CPU/GPU sur un modèle synthétique. Un venv GPU isolé expose bien
   CUDA/TensorRT, mais cuBLAS et la construction TensorRT refusent la compute
   capability 5.2 du GPU local. Le poste local Maxwell ne peut donc pas être la
   preuve d’un TensorRT moderne. `strict_providers=True` empêche désormais une
   fausse validation par fallback CPU.
4. [partiellement livré] Publier une matrice de compatibilité par OS, Python,
   runtime, architecture CPU/GPU et licence de checkpoint. `yase diagnostics`
   émet désormais les versions des distributions détectées et les providers
   optionnels ; la matrice multi-OS/hardware et les licences de checkpoints
   restent à exécuter dans des environnements de release dédiés.
5. [partiellement livré] Ajouter une procédure de migration pour les versions
   futures de résultats, bundles et checkpoints, avec fixtures de versions
   précédentes. Les aliases de résultats et checkpoints non versionnés sont
   maintenant migrés explicitement ; les migrations de versions majeures et
   fixtures historiques réelles restent à publier.

## Tâches P1 — production et exploitation

1. [partiellement livré] Ajouter une stratégie de cache explicite pour modèles
   et sessions, avec cycle de vie, fermeture et limites mémoire ;
   `RuntimeCache` fournit désormais un LRU borné thread-safe, réutilisable par
   TorchScript/ONNX/OpenVINO et par les ressources applicatives. Il ne
   télécharge rien et se ferme explicitement. La télémétrie mémoire détaillée
   et le partage optimisé des contextes TensorRT restent à traiter.
   `Yase.close()` relaie maintenant la fermeture des backends possédés.
2. Propager les contextes de trace dans les sinks, batches et appels de
   backends afin de relier une frame à ses stages.
3. [partiellement livré] Ajouter timeouts, quotas et annulation au boundary ASGI
   ; le pool de workers borné, `timeout_seconds`, réponse 504 et libération
   sûre des images sont désormais implémentés. L’authentification, le
   rate-limit global et les quotas par tenant restent idéalement au proxy ou à
   la plateforme.
4. [partiellement livré] Enrichir les checkpoints avec une politique de
   migration et une empreinte du modèle/configuration qui a produit l’état.
   `model_fingerprint` est maintenant écrit et vérifiable au chargement ; le
   calcul automatique d’empreinte depuis tous les formats d’artefacts reste
   optionnel et dépendant du modèle.
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
fallback des kernels locaux. Sur le poste de référence, la mesure du
15 septembre 2026 donne environ 30×/37× de gain IoU/NMS à 64 boîtes et
35×/78× à 256 boîtes. Il reste à instrumenter les runtimes complets, car leur
latence dépend du modèle, de la résolution, du provider et du matériel.

La frontière image est également durcie : les chemins Pillow et l’ASGI
vérifient les en-têtes, valident les payloads complets et convertissent les
décompressions dangereuses en erreurs d’entrée contrôlées.

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
