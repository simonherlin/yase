# Yase 0.66 — procédure de release multi-plateforme

Ce document transforme la matrice de compatibilité en commandes reproductibles.
Il ne réactive aucune pipeline hébergée : la release est lancée depuis un
poste Linux/macOS/Windows ou un runner externe contrôlé par l’équipe.

## Matrice de wheels

| Cible | Python | Distribution | Validation locale |
|---|---|---|---|
| Linux x86_64 | 3.10–3.13 | wheels natives manylinux | Docker/Podman + cibuildwheel |
| Linux arm64 | 3.10–3.13 | wheels natives manylinux | hôte arm64 ou émulation contrôlée |
| macOS x86_64 | 3.10–3.13 | wheel native macOS | macOS + Xcode CLT |
| macOS arm64 | 3.10–3.13 | wheel native macOS | Apple Silicon + Xcode CLT |
| Windows x86_64 | 3.10–3.13 | wheel native Windows | MSVC + Python ciblé |

Le profil cibuildwheel sélectionne explicitement CPython 3.10, 3.11, 3.12
et 3.13, conformément à `requires-python`. Une wheel portable
`py3-none-any` reste disponible avec `make build`; la wheel native ajoute
uniquement le kernel C++ optionnel IoU/NMS.

## Commandes

```bash
# Environnement release dédié
uvx cibuildwheel --version

# OS courant, architectures natives de la machine
make wheels

# Build local rapide (réutilise l'environnement uv déjà synchronisé)
make build

# Build PEP 517 isolé pour une validation de release
make build-isolated

# Linux x86_64 ciblé, un seul interpréteur pour déboguer
uv run python tools/build_wheels.py --platform linux \
  --only cp312-manylinux_x86_64

# macOS ou Windows : exécuter sur l’OS correspondant
uv run python tools/build_wheels.py --platform macos --archs native
uv run python tools/build_wheels.py --platform windows --archs native
```

La construction Linux nécessite Docker ou Podman ; macOS et Windows doivent
disposer de leur toolchain native. Une machine Linux ne prétend donc pas avoir
validé les wheels macOS/Windows. Chaque wheel native doit passer le smoke test
`NATIVE_AVAILABLE`; les wheels portables doivent passer les tests Python avec
le fallback pur Python.

## Runtime GPU : gates matériels séparés

Les wheels Yase ne contiennent ni CUDA, ni cuDNN, ni TensorRT, ni poids de
modèle. Avant d’annoncer une compatibilité GPU, le poste cible doit produire :

```bash
yase diagnostics --providers
uv run python tools/runtime_smoke.py
nvidia-smi                 # NVIDIA uniquement
nvcc --version             # si le toolkit CUDA est installé
trtexec --help             # installation TensorRT non-pip uniquement
```

ONNX Runtime GPU doit être installé avec un couple CUDA/cuDNN compatible avec
sa version ; OpenVINO doit rapporter le device réellement disponible ;
TensorRT doit être testé sur la famille de GPU cible avec l’engine correspondant.
Un plan TensorRT sérialisé n’est pas supposé portable entre plateformes ou
architectures GPU : produire une matrice par engine, OS, CUDA, driver et GPU.

## Sources officielles consultées

- cibuildwheel : <https://cibuildwheel.pypa.io/en/stable/options/>
- ONNX Runtime CUDA EP : <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- OpenVINO devices : <https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>
- TensorRT prerequisites : <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html>
- TensorRT support matrix : <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html>
