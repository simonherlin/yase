# Yase 0.62.0 — release readiness report

Date: 2026-09-15

## Gates executed

- `make check`: Ruff lint, format verification, and 142 tests pass on Python
  3.12.
- `pytest --cov=yase`: 82.38% total coverage on Python 3.9, above the 80%
  project gate.
- `uv build`: `yase-0.62.0-py3-none-any.whl` and
  `yase-0.62.0.tar.gz` build successfully.
- A fresh Python 3.9 environment installs the exact wheel with NumPy/Pillow,
  imports the public API, and round-trips a serialized semantic result.
- Wheel inspection contains the new config, runtime, serialization, and ASGI
  modules and no native binary; the sdist contains `native/yase_native.cpp` and
  the native builder.
- The repository remains clean after validation and every implementation cycle
  has a dedicated commit.

## What is covered

The release includes resilient image/batch/pipeline extraction, video and
realtime primitives, OpenVINO async queues, ONNX I/O binding, TensorRT runner
pools, Qdrant named vectors and payload indexes, versioned result JSON,
registry-controlled JSON/TOML configuration, optional OpenTelemetry, and the
dependency-free ASGI single/batch service.

## Explicit limits

The current development host has no live ONNX Runtime, OpenVINO, TensorRT,
Qdrant, Transformers, PaddleOCR, or CUDA provider installed. Those integrations
are contract-tested with fakes and lazy-import checks, but not hardware-tested
in this environment. Production CI must add provider-specific jobs and record
the supported driver/runtime matrix.

Yase never downloads model weights. Applications must distribute and verify
their artifacts, configure authentication/rate limiting at the service edge,
and select an appropriate TensorRT context pool size for available GPU memory.

See the [full 0.62 architecture audit](full-package-audit.md) for the next
tasks: live runtime CI, trace propagation through every stage, checkpoint
migrations, and keyframe/VLM routing.
