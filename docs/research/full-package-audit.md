# Yase — audit complet du package et plan de production

Date de l'audit : 2026-09-15  
État inspecté : `0.58.0`

Ce document est la référence de pilotage technique. Il distingue ce qui est
déjà livré, ce qui est contractuellement couvert mais non testé sur matériel,
et ce qui manque encore pour prétendre à une utilisation mondiale en
production.

## 1. Conclusion exécutive

Yase possède déjà une base saine et inhabituelle pour un package léger : une
API Python stable, un résultat sémantique extensible, des adaptateurs lazy, des
pipelines/DAG, de la vidéo fichier et temps réel, du tracking, de la mémoire
temporelle, de l'identité cross-camera, de la recherche vectorielle, de la
sérialisation, des métriques, une CLI et une accélération C++ facultative.

Le package n'est toutefois pas encore une plateforme universelle complète.
Les principaux risques ne sont plus l'absence d'un modèle supplémentaire, mais
les frontières de production : exécution asynchrone réelle sur accélérateur,
tests d'intégration par runtime, gestion explicite des schémas de vecteurs,
observabilité distribuée, API de service, reproductibilité des modèles et
matrice de compatibilité.

Décision d'architecture : conserver un cœur NumPy/Pillow sans framework lourd,
et placer chaque capacité coûteuse derrière un adaptateur explicite. Les
optimisations doivent être ajoutées au niveau du runtime ou du batch, sans
polluer `SemanticResult`, `ObservationBundle` ni les contrats de tracking.

## 2. Inventaire par sous-système

| Sous-système | État actuel | Niveau | Risque restant |
|---|---|---:|---|
| Entrées image/Pillow/NumPy | `load_image`, limites pixels/bytes/canaux | livré | formats vidéo et métadonnées EXIF à formaliser |
| Résultat sémantique | profondeur, segmentation, détections, OCR, VLM, embeddings, relations, événements | livré | version majeure et champs multimodaux à stabiliser |
| Extraction unitaire | `Yase.extract`, callable, mapping, array | livré | métriques par backend à enrichir |
| Batch image | `Yase.extract_many`, backends natifs, fallback parallèle, skip partiel | livré | erreurs de provider à mesurer par runtime |
| Pipeline composite | fusion de champs, conflits, `extract_batch` | livré | fusion probabiliste et calibration avancée |
| Scheduler DAG | dépendances, cache borné, async, deadline, cancellation | livré | exécution durable/distribuée absente |
| Vidéo fichier | stride, FPS, batch, ordre causal, sinks | livré | seek/reprise de capture non persistée |
| Vidéo temps réel | worker latest-frame/backpressure | livré | queue multi-stage et cancellation plus fine |
| Tracking | IoU, ByteTrack-like, tracker externe | livré | BoT-SORT/DeepSORT officiels restent intégrations externes |
| Mémoire / ReID | EMA, identité cross-camera, checkpoints | livré | index ReID scalable et calibration multi-caméra |
| Événements | présence, zone, ligne, dwell | livré | fenêtres temporelles complexes et CEP |
| OCR | Tesseract, PaddleOCR lazy | livré | batch réel et document layout avancé |
| VLM | Transformers, JSON schema subset, batch | livré | streaming tokens, vidéo native, contraintes JSON avancées |
| Runtime ONNX | batch, providers, graph options, I/O binding optionnel | livré | vraie validation GPU/EP en CI |
| Runtime OpenVINO | sync CPU/GPU/NPU/AUTO, `AsyncInferQueue` ordonné | livré | vraie validation hardware en CI |
| Runtime TensorRT | plan et runner custom | livré | buffers CUDA réutilisables, context pools, shapes dynamiques |
| Retrieval local | index NumPy, NPZ, namespace d'embedding | livré | HNSW/FAISS local optionnel |
| Retrieval Qdrant | upsert/query, filtres, namespace, named vectors, payload indexes | livré | migration de schémas multi-vecteurs |
| Observabilité | métriques thread-safe, JSON, Prometheus text, OpenTelemetry spans | livré | propagation de trace dans tous les stages |
| Packaging | wheel pure portable, sdist C++/builder, extras lazy | livré | matrice OS/Python/accélérateur à publier |
| Native C++ | IoU/NMS hot paths, fallback Python | livré | ABI/build wheels spécialisés non distribués |
| CLI | image, vidéo, benchmark, diagnostics, catalogues, évaluations | livré | config déclarative et gRPC éventuels |
| Service | ASGI borné, health/readiness, métriques, extraction base64 | livré | auth/rate-limit laissés à l'infrastructure |
| Documentation | README/API/architecture/recherche/status/release readiness | livré | guides d'intégration runtime à enrichir |

## 3. Recherche technologique vérifiée

### Vision-language

La documentation Transformers officielle utilise `AutoProcessor` et
`AutoModelForImageTextToText`, la génération puis `batch_decode`; elle décrit
également les conversations multimodales et l'usage de `generate`.
[Documentation officielle image-text-to-text](https://huggingface.co/docs/transformers/tasks/image_text_to_text)

Conséquence pour Yase : l'adaptateur doit conserver une séparation entre
préparation conversationnelle, génération et parsing structuré. Le batch doit
regrouper les entrées homogènes, mais ne doit jamais mélanger les schémas de
sortie sans les associer à leur requête.

### ONNX Runtime

ONNX Runtime documente trois niveaux d'optimisation de graphe et un mode
offline pouvant sérialiser le graphe optimisé.
[Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

Pour les providers non CPU, la documentation recommande I/O binding afin de
placer les entrées/sorties sur le device et d'éviter les copies implicites.
[I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)

Yase expose maintenant `use_io_binding=True`; la prochaine étape doit
ajouter des tests hardware-gated avec CUDA/TensorRT Execution Provider et
mesurer séparément preprocessing, copie, kernel et postprocessing.

### OpenVINO

OpenVINO fournit `AsyncInferQueue`, un pool de requêtes avec
`start_async`, callbacks et `wait_all`.
[AsyncInferQueue officiel](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.AsyncInferQueue.html)

L'adaptateur Yase conserve l'ordre des résultats par `userdata` et n'expose
jamais les objets de requête OpenVINO dans le contrat public. Il reste à
ajouter un smoke test sur une vraie cible CPU/GPU/NPU.

### TensorRT

L'API Python officielle repose sur un execution context, des adresses de
tenseurs et un CUDA stream; plusieurs contexts permettent le recouvrement,
mais un même context ne doit pas être utilisé concurremment.
[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)

La couche Yase doit donc gérer un pool de contexts/streams, ou déléguer cette
responsabilité à un runner externe clairement documenté. Il ne faut pas
simuler une concurrence thread-safe avec un seul context partagé.

### Qdrant

Qdrant supporte des named vectors, permettant plusieurs espaces et métriques
dans un même point, et les payload indexes pour accélérer les filtres.
[Collections et named vectors](https://qdrant.tech/documentation/manage-data/collections/)
[Payload filtering et indexes](https://qdrant.tech/documentation/search/filtering/)

Yase lie déjà un index à un espace logique et supporte explicitement les deux
stratégies : collection historique mono-vecteur ou `vector_name` Qdrant nommé.
`payload_indexes` rend les champs filtrés indexables. Le champ `space` dans le
payload reste une provenance applicative et ne remplace pas une validation de
schéma vectoriel côté serveur.

### Détection temps réel

RF-DETR, Grounding DINO, SAM3 et les familles Transformers sont correctement
placés derrière des adaptateurs et des notices de licence. Les poids ne sont
jamais téléchargés implicitement. Les intégrations doivent rester optionnelles
et leurs performances doivent être mesurées sur le matériel cible.

## 4. Backlog priorisé

### P0 — avant une release production générale

1. Construire une matrice CI de smoke tests optionnels : Python 3.9–3.13,
   Linux/macOS/Windows, CPU, CUDA/ONNX EP, OpenVINO si disponible.
2. Ajouter des tests hardware-gated qui vérifient réellement les providers,
   les formes dynamiques, les erreurs OOM et la fermeture des ressources.
3. Versionner explicitement `SemanticResult`, `ObservationBundle` et les
   checkpoints avec politique de migration et tests de compatibilité.
4. Ajouter une politique de provenance obligatoire pour les sorties de modèle
   dans les flux durables : identifiant, révision, hash et configuration.
5. Documenter les limites de sûreté : taille, mémoire, nombre de frames,
   profondeur JSON, timeouts et comportement en cas de saturation.

### P1 — fonctionnalités de plateforme

1. Étendre TensorRT avec pool de contexts, buffers réutilisables et streams.
2. Propager les contextes OpenTelemetry dans les stages, batches et backends.
3. Ajouter un format de configuration déclaratif (TOML/YAML optionnel) qui
   instancie registry, pipeline, limites, sinks et checkpoints.
4. Ajouter une stratégie keyframe/VLM pour éviter d'appeler un VLM lourd à
   chaque frame vidéo.

### P2 — extension mondiale

1. FAISS/HNSW local optionnel et abstraction de distance non limitée à cosine.
2. Document understanding : layout, tables, formulaires et pages PDF.
3. Audio/ASR et synchronisation audio-vidéo si le périmètre devient multimodal.
4. Connecteurs Kafka, Redis Streams et stockage objet via plugins séparés.
5. Packs de modèles versionnés et benchmarks reproductibles publics.

## 5. Ce qui a été livré dans les derniers cycles

- Batch vidéo avec ordre causal et gestion des erreurs.
- Génération VLM batchée et sorties JSON structurées.
- Composite multi-backends batché.
- Checkpoints atomiques unifiés et intégration directe aux streams.
- Provenance et filtrage d'espace pour NumPy/Qdrant.
- CLI complète pour les backends du registry.
- ONNX I/O binding optionnel.
- Comptage correct des pertes en temps réel.
- OpenVINO `AsyncInferQueue` et résultat ordonné.
- OpenTelemetry optionnel et service ASGI borné.
- Qdrant named vectors et indexes payload explicites.
- Pipeline batch tolérant aux erreurs partielles.
- Wheel pure portable et sources natives conservées dans le sdist.

## 6. Critères de sortie d'une release 1.x

Une release production doit satisfaire simultanément :

- tests unitaires et contractuels sur toutes les interfaces publiques ;
- au moins un smoke test réel par runtime installé ;
- aucune dépendance lourde importée par `import yase` ;
- résultats ordonnés et identifiables sur image, batch, vidéo et temps réel ;
- checkpoints testés sur restart et migration de version ;
- métriques avec corrélation de requête et absence de cardinalité non bornée ;
- wheel/sdist installables depuis un environnement propre ;
- documentation des licences, modèles, limites et incompatibilités ;
- benchmark séparant prétraitement, transfert, inférence, post-traitement et
  latence de bout en bout.

## 7. Décision immédiate

Les tâches suivantes sont les plus rentables et doivent être implémentées dans
cet ordre :

1. tester les extras sur les runtimes et matériels réellement supportés ;
2. ajouter un pool TensorRT de contexts et buffers réutilisables ;
3. versionner/migrer explicitement les schémas `SemanticResult` et checkpoints ;
4. produire une configuration déclarative et une stratégie keyframe/VLM ;
5. publier des benchmarks séparant preprocessing, transferts, kernels et
   post-traitement.

Le projet ne doit pas intégrer de framework obligatoire supplémentaire tant que
ces frontières de production ne sont pas stabilisées.
