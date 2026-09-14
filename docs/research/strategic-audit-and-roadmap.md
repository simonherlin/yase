# Yase — audit stratégique et roadmap d’industrialisation

## Résumé exécutif

Yase dispose déjà d’une base inhabituelle pour un prototype : un cœur
NumPy/Pillow léger, des adaptateurs TorchScript/ONNX/OpenVINO/TensorRT, des
pipelines, du tracking, de la mémoire temporelle, des événements, des
identités cross-camera, de la recherche vectorielle, des métriques HOTA/mAP et
des lecteurs MOT/COCO. La version de départ de cet audit était `0.12.0`, avec
82 tests et une
couverture supérieure à 80 %.

Le risque principal n’est donc plus le manque de fonctionnalités isolées. Le
risque est la fragmentation : plusieurs abstractions portent une information
semblable mais pas identique, les sorties de modèles ne sont pas toujours
converties vers un graphe sémantique commun, et les chemins image/batch/video/
live n’ont pas encore un ordonnanceur partagé. Une nouvelle classe de modèle
ne doit pas seulement produire un tableau : elle doit déclarer ses capacités,
son coût, sa provenance, sa précision, ses limites et son comportement
temporel.

La stratégie recommandée est de faire de Yase un **runtime d’orchestration
sémantique multimodal**, et non un énième dépôt de poids ou une copie de
framework de détection. Le produit central devient :

1. un graphe d’observations versionné et typé ;
2. un scheduler de stages avec budgets, annulation, backpressure et cache ;
3. des adaptateurs interchangeables pour les meilleurs modèles disponibles ;
4. un protocole d’évaluation et de provenance reproductible ;
5. une surface de déploiement allant du notebook CPU au flux multi-caméras
   accéléré.

Cette direction permet d’utiliser un détecteur temps réel pour chaque frame,
SAM 3 pour les concepts ouverts, un encodeur DINOv2/SigLIP pour la recherche,
PaddleOCR pour les documents et un VLM pour les questions complexes sans
forcer tous les utilisateurs à installer tous les frameworks.

## État réel du dépôt

### Ce qui est déjà solide

- Le cœur ne force pas PyTorch, CUDA, ONNX Runtime, OpenVINO ou TensorRT.
- `SemanticResult` conserve les sorties classiques : profondeur, segmentation,
  détections, OCR, captions, scène, embeddings, événements et métadonnées.
- Les backends lourds sont lazy-loaded et n’ont pas de téléchargement implicite
  de poids.
- Les chemins séquentiel, batch, vidéo et live existent déjà et ont des tests
  fake déterministes.
- Le registre, les cartes de modèles et les helpers d’artefacts donnent une
  base pour la provenance et les licences.
- MOT HOTA, COCO-style AP/mAP, masques, export COCO et SHA-256 rendent déjà
  possibles des quality gates locaux.
- Le projet possède une CI multi-version Python et des smoke tests de wheel.

### Ce qui est seulement partiellement résolu

| Domaine | Niveau actuel | Risque concret |
|---|---:|---|
| Contrat sémantique | 5/10 | Pas de types natifs pour keypoints, pose, profondeur calibrée, trajectoires riches, régions orientées ou relations. |
| Orchestration | 4/10 | Pipeline séquentiel ; pas de DAG async, de budget par stage, de cancellation ni de cache de résultats. |
| Détection | 6/10 | Adaptateurs présents, mais post-traitement, NMS, calibration et conversion des formats restent modèle-dépendants. |
| Segmentation | 5/10 | Masques disponibles, mais pas de RLE/polygone unifié dans le contrat et pas de tracking mask-first. |
| Tracking | 5/10 | Baselines IoU/ByteTrack-lite ; BoT-SORT/OC-SORT/Deep-OC-SORT ne sont pas intégrés derrière une API officielle. |
| VLM | 4/10 | Adapter générique, mais pas de sortie structurée contrainte, cache, keyframes, tool calls ou garde-fous. |
| Embeddings | 5/10 | Index NumPy/Qdrant, mais provenance d’espace, version, dimension et multimodalité ne sont pas contractuelles. |
| OCR/documents | 4/10 | Tesseract/PaddleOCR ; pas de document graph, orientation, tables, layout et CER/WER complet. |
| Runtime | 5/10 | Adapters présents ; manque cache engine, I/O binding, quantification, warm-up et matrices hardware. |
| Évaluation | 6/10 | HOTA/mAP utiles, mais pas protocole multi-tâches versionné ni intégration COCO officielle avec area/maxDets/ignore. |
| Production | 3/10 | Pas d’OpenTelemetry effectif, health checks, métriques Prometheus, service HTTP, persistence de jobs ou isolation. |
| Distribution | 5/10 | Wheel minimal et extras ; pas de matrice d’images Docker, extras par plateforme ou policy de compatibilité runtime. |

### Dette technique prioritaire

1. La liste `detections` est parfois typée, parfois arbitraire, selon le
   backend. Le contrat doit être configurable mais les modes sûrs doivent
   exister.
2. `SemanticResult` mélange des valeurs de frame, de track et de dataset sans
   distinguer leur portée. Une observation frame ne devrait pas être confondue
   avec une identité persistante ou une assertion VLM.
3. Le temps est un `float` et les erreurs sont souvent des exceptions simples.
   Les pipelines longs ont besoin d’un identifiant de frame, d’un timestamp
   monotone, d’une source et d’un statut explicite.
4. Les runtimes sont testés avec des fakes, mais pas avec des petits modèles
   ONNX/OpenVINO/TensorRT réellement exécutables dans CI hardware-gated.
5. Les métriques locales sont utiles mais ne doivent pas être vendues comme
   l’implémentation officielle COCO. Les paramètres évalués doivent toujours
   être enregistrés avec le résultat.
6. Le dépôt n’a pas de fixtures d’images/vidéos versionnées, de golden outputs
   ni de budget de régression de latence.

## Résultats de la recherche technologique

### Détection et segmentation ouvertes

SAM 3 est présenté par Meta comme un modèle unifié qui détecte, segmente et
tracke les instances correspondant à des prompts texte, exemplaires visuels ou
combinaisons des deux.^1 La publication décrit le problème de Promptable
Concept Segmentation et un identifiant par instance dans l’image ou la vidéo.^2
La conclusion d’architecture est claire : SAM 3 doit être un stage coûteux de
recherche/annotation et non le détecteur de chaque frame.

Grounding DINO reste le stage approprié quand l’application fournit un texte,
une expression référente ou une catégorie hors vocabulaire fixe ; son travail
fondateur traite explicitement l’open-set object detection.^3 RF-DETR représente
une autre voie pour le closed-set temps réel et annonce aussi des variantes
detection/segmentation/keypoints dans son dépôt actif.^4

**Décision Yase :** séparer trois profils, avec des contrats identiques :

- `fast_closed_set` : détecteur/export ONNX ou TensorRT pour chaque frame ;
- `open_vocabulary` : Grounding DINO ou équivalent, déclenché par prompt ;
- `concept_segmentation` : SAM 3, déclenché par keyframe, interaction ou
  événement.

### Tracking

ByteTrack part de l’idée d’associer aussi les détections de score faible, qui
peuvent correspondre à des objets occlus ; le papier rapporte des gains
d’IDF1 avec cette seconde association.^5 BoT-SORT ajoute des signaux
d’apparence et de compensation du mouvement caméra.^6 OC-SORT remplace une
partie de la confiance excessive dans une prédiction linéaire par une approche
centrée sur les observations, particulièrement intéressante lors des
occlusions et mouvements non linéaires.^7 Deep OC-SORT ajoute une ré-identi-
fication adaptative et une compensation caméra à cette famille.^8

**Décision Yase :** ne pas réimplémenter tous les dépôts à l’identique dans le
core. Ajouter une API `Tracker` complète avec : `update`, `predict`, `reset`,
`state_dict`, `load_state_dict`, `capabilities`, et implémenter :

1. baseline NumPy actuelle ;
2. ByteTrack officiel derrière extra ;
3. OC-SORT/Deep OC-SORT pour mouvement non linéaire et apparence ;
4. BoT-SORT pour caméra mobile et scènes denses.

Chaque tracker doit déclarer s’il consomme box, mask, embedding ou optical
flow, et le benchmark doit comparer HOTA, AssA, DetA, IDF1, MOTA et latence.

### Encoders et recherche multimodale

DINOv2 fournit des features visuelles auto-supervisées, avec des variantes de
dimension 384 à 1536, destinées entre autres à la classification, la
segmentation, la profondeur et la recherche par voisins.^9 SigLIP sépare les
encodeurs image et texte et utilise une perte sigmoid paire-à-paire, ce qui le
rend adapté à la recherche image-texte et au zero-shot.^10

**Décision Yase :** `Embedding` doit porter `space`, `model_id`, `revision`,
`dimension`, `normalized`, `modality` et `timestamp`. Un vecteur DINOv2 ne doit
pas être mélangé silencieusement avec un vecteur SigLIP dans un index. Ajouter
un `EmbeddingRecord` et des namespaces d’index.

### OCR et documents

PaddleOCR documente une couverture multilingue large et son évolution vers des
pipelines de document parsing ; PaddleOCR 3 présente notamment PP-OCRv5,
PP-StructureV3 et PP-ChatOCRv4.^11 docTR expose un pipeline OCR en deux étapes,
localisation puis reconnaissance.^12 Tesseract reste un fallback local simple.

**Décision Yase :** ajouter `DocumentPage`, `TextLine`, `TableRegion` et
`DocumentGraph` optionnels. Les sorties OCR doivent conserver la langue, le
script, la rotation, le polygon, la confidence de détection et la confidence
de reconnaissance séparément. Évaluer CER, WER, word accuracy, reading order
et table cell accuracy.

### Runtime et hardware

ONNX Runtime expose une même API Python pour de nombreux Execution Providers,
dont CPU, CUDA, TensorRT et OpenVINO, et alloue les nœuds aux providers
capables de les exécuter.^13 La documentation TensorRT recommande de configurer
les adresses de tenseurs puis d’appeler `execute_async_v3` avec un stream CUDA.^14
OpenVINO prend en charge CPU, GPU et NPU, ainsi que AUTO, HETERO et le batching
automatique, mais ses capacités varient par appareil et par version.^15

**Décision Yase :** traiter le runtime comme un objet de capacité, pas comme
un simple nom. Le benchmark doit enregistrer :

- OS, Python, architecture CPU, GPU/NPU et versions driver ;
- provider réellement actif, fallback éventuel et précision ;
- warm-up, p50/p95/p99, throughput, mémoire, power si disponible ;
- shape dynamique, batch, transfert host/device et I/O binding ;
- hash de l’artefact et configuration complète.

Un résultat CPU ne doit jamais être présenté comme une promesse GPU.

### VLM et sortie structurée

Les APIs Transformers récentes exposent un chemin commun image-text-to-text
pour les VLM, avec captioning, question answering et sorties ouvertes.^16 Cela
est puissant mais non déterministe et coûteux. Le VLM doit donc être appelé par
un scheduler sur keyframes, événements, requêtes utilisateur ou faibles
confiances ; il ne doit pas remplacer la perception persistante.

**Décision Yase :** ajouter un `StructuredQuery` avec prompt, schéma JSON,
budget de tokens, timeout, fréquence maximale, cache key et politique
d’abstention. Valider la sortie avec un schéma strict, conserver le prompt et
le hash de l’image, et distinguer `assertion`, `hypothesis` et `caption`.

## Architecture cible

```text
InputSource
  ├── ImageSource / DirectorySource / VideoSource / CameraSource
  └── Decode + timestamp monotone + frame_id + source_id
        │
        ▼
ObservationScheduler
  ├── bounded queues + backpressure + cancellation
  ├── budget CPU/GPU/mémoire/latence
  ├── cache de preprocess et de résultats
  ├── keyframe/event policy
  └── stage graph / fan-out / fan-in
        │
        ├── Perception: detector, segmenter, depth, pose, OCR
        ├── Association: tracker, ReID, cross-camera identity
        ├── Semantics: VLM, tags, scene, relations
        ├── Retrieval: embeddings + index namespace
        └── Analytics: events, rules, aggregates
        │
        ▼
+SemanticObservation
  ├── frame observations
  ├── track observations
  ├── scene assertions
  ├── provenance + uncertainty + timings
  └── serializers: JSONL / COCO / MOT / Parquet / sinks
```

Le core actuel doit évoluer progressivement vers ce modèle, sans casser
`SemanticResult`. Une migration compatible peut faire de `SemanticResult` une
vue de frame sur un `ObservationBundle` plus riche.

## Backlog priorisé

### P0 — contrats et fiabilité

1. Ajouter `FrameRef(frame_id, source_id, timestamp, width, height, color_order)`.
2. Ajouter `ObservationBundle` avec portée `frame`, `track`, `scene`.
3. Ajouter `Uncertainty` et séparer score de détection, probabilité calibrée
   et qualité de masque.
4. Ajouter `ModelProvenance` typé : id, revision, artifact hash, license,
   runtime, device, precision.
5. Ajouter un protocole `Stage` avec `capabilities`, `cost`, `requires`,
   `provides`, `reset` et `close`.
6. Faire évoluer `SemanticPipeline` vers un DAG validé avec détection de
   dépendances et collisions de champs.
7. Ajouter une politique d’erreur structurée : retry, skip, fallback,
   abstain, dead-letter.
8. Ajouter fixtures tiny images/video et golden JSON versionnés.

### P1 — scheduler image/video/live

9. Créer `ObservationScheduler` sync et async avec bounded queues.
10. Implémenter budget `max_latency_ms`, `max_gpu_memory_mb`, `max_vlm_calls`
    et `max_fps`.
11. Ajouter cancellation et fermeture idempotente de toutes les queues.
12. Ajouter cache LRU de preprocessing et cache de stages par image hash +
    model revision + configuration.
13. Ajouter keyframe policies : interval, scene change, motion, event, novelty.
14. Ajouter fan-out/fan-in pour detector + depth + OCR et mesurer chaque stage.
15. Exposer queue depth, dropped frames et saturation dans `VideoStats`.
16. Ajouter un sink JSONL, callback, queue Python et callback async.

### P1 — contrats de vision riches

17. `MaskInstance` avec format polygon/RLE/bitmap, size, area et provenance.
18. `Keypoints` avec noms, coordonnées, visibilité et score.
19. `OrientedBoundingBox` et conversion xyxy/xywh/rotated.
20. `DepthMap` avec scale, invalid value, metric/relative et camera metadata.
21. `EmbeddingRecord` avec namespace et provenance.
22. `Relation(subject, predicate, object, score, evidence)` pour scene graphs.
23. `TextLine`, `DocumentPage`, `TableRegion`, reading order et language.
24. Compatibilité de serialization et export pour chaque type.

### P1 — adapters prioritaires

25. Intégrer un vrai adapter DINOv2 local et batch-capable.
26. Intégrer SigLIP image/text avec zero-shot classification et retrieval.
27. Ajouter un Grounding DINO réellement normalisé avec prompt cache.
28. Renforcer SAM 3 image/video : instances, IDs, prompts et mask provenance.
29. Ajouter docTR derrière une extra OCR-document distincte.
30. Mettre à jour PaddleOCR pour les API 3.x en conservant le fallback 2.x.
31. Ajouter ByteTrack/OC-SORT/BoT-SORT via `ExternalTrackerAdapter` testable.
32. Ajouter un depth adapter métrique avec calibration intrinsics/extrinsics.

### P2 — runtime production

33. ONNX I/O binding CPU/CUDA et gestion explicite des fallbacks.
34. OpenVINO model cache, compiled model reuse et device query.
35. TensorRT engine cache, profiles dynamiques, FP16/INT8 metadata.
36. Ajouter ExecuTorch export/inference derrière extra mobile.
37. Ajouter une matrice de compatibilité runtime/Python/OS/architecture.
38. Ajouter des tests hardware-gated et des fixtures ONNX minuscules en CI.
39. Ajouter warmup, concurrency, batch auto et throughput benchmark.
40. Ajouter validation d’input shape/layout/dtype avant le premier appel.

### P2 — évaluation scientifique

41. Wrap officiel `pycocotools` quand installé, avec résultat explicitement
    marqué `official=True`.
42. Ajouter COCO area ranges, maxDets et ignore/crowd sans les simuler.
43. Ajouter LVIS long-tail metrics et category-frequency breakdown.
44. Ajouter MOTChallenge official adapter et HOTA/AssA/DetA auditables.
45. Ajouter OCR CER/WER et document/table metrics.
46. Ajouter retrieval Recall@K, mAP@K, nDCG et cross-modal tests.
47. Ajouter calibration ECE, Brier, reliability bins et threshold search.
48. Ajouter robustness suites : blur, noise, scale, crop, lighting, occlusion.
49. Ajouter golden regression tolerance par dtype/runtime.
50. Exporter benchmark records en JSONL et Parquet optionnel.

### P2 — production et sécurité

51. Instrumenter OpenTelemetry spans par source/stage/model/provider.
52. Ajouter Prometheus-compatible counters/histograms sans dépendance core.
53. Ajouter health/readiness checks et diagnostic provider.
54. Ajouter limits de pixels, durée vidéo, mémoire et nombre de detections.
55. Ajouter protection contre images corrompues, decompression bombs et payload
    VLM trop grands.
56. Ajouter redaction configurable pour OCR/PII et logs sans données sensibles.
57. Ajouter policy de licence par modèle et manifest SPDX.
58. Ajouter SBOM, provenance wheel, signatures et hash lockfile.
59. Ajouter images Docker CPU/ONNX, CUDA/TensorRT et OpenVINO séparées.
60. Ajouter service HTTP/gRPC optionnel sans polluer le package core.

## Plan d’exécution des prochains cycles

### Cycle A — observation contract

Implémenter les types `FrameRef`, `ModelProvenance`, `Uncertainty`,
`ObservationBundle`, `MaskInstance`, `EmbeddingRecord` et les serializers. Le
critère de sortie est une migration sans changement de comportement pour les
82 tests existants.

### Cycle B — scheduler

Implémenter le DAG, le cache et les budgets avec des stages fake. Tester ordre,
backpressure, cancellation, timeout, fallback et fermeture sous exception.

### Cycle C — perception réelle

Ajouter DINOv2/SigLIP/Grounding DINO et renforcer SAM 3. Chaque adapter doit
avoir une fixture injectée, un mode local-only, provenance complète, batch et
contract tests communs.

### Cycle D — tracking et vidéo

Ajouter OC-SORT/BoT-SORT derrière le même protocole, puis keyframes, ReID et
camera motion compensation. Comparer HOTA/IDF1/latence sur fixtures MOT.

### Cycle E — production

OpenTelemetry, metrics, limits, diagnostics, manifests de licence, Docker et
service optionnel. Tester les pannes, les grosses images et les erreurs de
provider avant toute promesse de production.

## Décisions à ne pas prendre

- Ne pas télécharger automatiquement des poids dans un import ou un
  constructeur par défaut.
- Ne pas faire d’Ultralytics une dépendance core sans stratégie de licence ; sa
  licence par défaut et ses conditions commerciales doivent rester une décision
  explicite de déploiement.^17
- Ne pas appeler une métrique locale « COCO mAP officiel » sans passer les
  paramètres et la version de `pycocotools`.
- Ne pas exécuter un VLM lourd à chaque frame par défaut.
- Ne pas mélanger des embeddings de modèles différents dans le même espace.
- Ne pas cacher un fallback GPU vers CPU : il doit apparaître dans les
  métadonnées et dans les métriques.
- Ne pas promettre une latence sans hardware, résolution, batch et précision
  documentés.

## Définition de “meilleur package”

Yase sera meilleur non pas lorsqu’il contiendra le plus de classes Python, mais
lorsqu’un utilisateur pourra :

1. brancher une image, une vidéo ou une caméra avec le même contrat ;
2. choisir un backend rapide, ouvert ou génératif sans réécrire son application ;
3. obtenir des observations typées avec provenance et incertitude ;
4. dégrader proprement quand un runtime ou un modèle est absent ;
5. expliquer pourquoi un stage a été exécuté ou sauté ;
6. reproduire les métriques avec les mêmes artefacts et paramètres ;
7. observer les coûts, les erreurs, les frames perdues et la qualité ;
8. exporter les résultats vers COCO, MOT, JSONL, Parquet ou un index vectoriel.

## Sources

1. Meta AI, [Introducing Meta Segment Anything Model 3](https://ai.meta.com/research/sam3/).
2. Carion et al., [SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), 2025.
3. Liu et al., [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023.
4. Roboflow, [RF-DETR repository and documentation](https://github.com/roboflow/rf-detr).
5. Zhang et al., [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), 2022.
6. Aharon et al., [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651), 2022.
7. Cao et al., [Observation-Centric SORT](https://arxiv.org/abs/2203.14360), 2022.
8. Du et al., [Deep OC-SORT](https://arxiv.org/abs/2302.11813), 2023.
9. Meta AI, [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).
10. Hugging Face, [SigLIP documentation](https://huggingface.co/docs/transformers/model_doc/siglip).
11. PaddlePaddle, [PaddleOCR documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/index.html).
12. Mindee, [docTR repository](https://github.com/mindee/doctr).
13. ONNX Runtime, [Execution Providers](https://onnxruntime.ai/docs/execution-providers/).
14. NVIDIA, [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).
15. Intel, [OpenVINO supported devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html).
16. Hugging Face, [Image-text-to-text tasks](https://huggingface.co/docs/transformers/main/tasks/image_text_to_text).
17. Ultralytics, [Licensing](https://www.ultralytics.com/license).
