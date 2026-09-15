# Yase 0.65.0 — release readiness

Validation date: 2026-09-15. Reference environment: Python 3.12.3 on Ubuntu
24.04.5 LTS, x86_64.

## Gates

| Gate | Result |
| --- | --- |
| Python policy | pass: 3.10–3.13 supported, 3.12 reference |
| Lockfile | pass: `uv lock --check` |
| Dependencies | pass: `uv pip check` |
| Ruff | pass: lint and format checks |
| Core suite | pass: 147 tests |
| Coverage | pass: above the configured 80% threshold |
| Runtime smoke | pass: OpenVINO CPU sync/async and ONNX CPU/I/O binding |
| Native extension | pass: C++17 IoU/NMS build and import |
| Distribution | pass: wheel and sdist build |
| Portable install | pass: clean Python 3.12 wheel environment |

## Delivered production hardening

- Image dimensions are checked from Pillow headers before conversion, while
  final decoded byte limits remain enforced.
- ASGI deployments can apply the same decoded-image limits to custom backends.
- `yase diagnostics --providers` reports actual ONNX Runtime providers,
  OpenVINO devices, PyTorch CUDA state, and TensorRT availability.
- Native backend batches emit `yase.extract_batch` tracing spans.
- Observation bundles can be reconstructed with `observation_from_dict()`;
  future schemas and non-lossless tensor summaries are rejected.
- Runtime CI creates temporary real ONNX/OpenVINO graphs and never downloads
  model weights.
- The RF-DETR extra follows its current upstream dependency contract.

## Hardware limitation

The reference host exposes an NVIDIA Quadro M3000M (compute capability 5.2)
and no CUDA toolkit. Current TensorRT documentation requires Turing or newer
GPUs and a matching CUDA toolkit, so TensorRT/CUDA execution is not claimed on
this workstation. OpenVINO CPU/GPU/AUTO and ONNX Runtime CPU are the validated
local paths. See the [runtime matrix](runtime-matrix-0.64.md) for exact
provider versions and observations.

## Reproduction

```bash
uv sync --locked --group runtime-test
uv run python tools/runtime_smoke.py
uv lock --check
uv pip check --python .venv/bin/python
make check
uv build
```

The generic wheel stays ABI-independent; the optional C++ extension is built
separately for the target Python ABI and platform.
