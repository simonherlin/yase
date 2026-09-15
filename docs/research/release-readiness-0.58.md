# Yase 0.58.0 — release readiness report

Date: 2026-09-15

## Validated gates

- `make check`: Ruff lint, Ruff format check, and 139 deterministic tests pass.
- `pytest --cov=yase`: 82.52% total coverage, above the configured 80% gate.
- `uv build`: source distribution and pure-Python wheel built successfully.
- Wheel inspection: all `yase` modules, `py.typed`, metadata, license, and CLI
  entry point are present; no accelerator binary is accidentally embedded.
- Sdist inspection: source, native C++17 fallback/extension source, tests,
  examples, documentation, and lockfile are present.
- Fresh Python 3.9 virtual environment: wheel installs with NumPy/Pillow and
  imports the public API; a callable extraction smoke test returns the expected
  tensor shape.

## Compatibility boundary

| Surface | Base install | Optional dependency | Hardware/host caveat |
| --- | --- | --- | --- |
| NumPy/Pillow extraction | yes | — | CPU-safe |
| ONNX Runtime | no | `yase[onnx]` | provider-specific |
| OpenVINO sync/async | no | `yase[openvino]` | CPU/GPU/NPU availability |
| TensorRT | no | `yase[tensorrt]` plus CUDA/Torch | NVIDIA CUDA required for native path |
| OCR/Transformers/video | no | corresponding extras | model weights remain application-managed |
| Qdrant | no | `yase[retrieval]` | server or local Qdrant client |
| OpenTelemetry | no | `yase[observability]` | exporter/provider configured by host |
| ASGI service | yes | ASGI server installed by host | deployment server is intentionally separate |
| Native C++ acceleration | no binary in wheel | local build from sdist | compiler and ABI are host-specific |

## Known limitations

The current environment does not contain ONNX Runtime, OpenVINO, TensorRT,
Qdrant, Transformers, PaddleOCR, or OpenTelemetry, so those integrations are
validated with injectable deterministic doubles and lazy-import checks rather
than live provider hardware. A release CI matrix should add jobs with each
extra and the provider versions used by supported deployments.

The package intentionally does not download model weights. Applications must
pin, distribute, and verify their model artifacts using `inspect_artifact()`
and `verify_artifact()` before production inference.

## Reproduction commands

```bash
uv lock
make check
uv run pytest --cov=yase --cov-report=term-missing -q
uv build
```

This report records package validation; the broader algorithm and production
roadmap remains in [the full package audit](full-package-audit.md).
