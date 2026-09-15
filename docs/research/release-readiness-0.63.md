# Yase 0.63.0 — release readiness report

Date: 2026-09-15

## Gates executed

- `make check`: Ruff lint, format verification, and **143 tests** pass on
  Python 3.12.
- `pytest --cov=yase`: **82.38%** total coverage on Python 3.9, above the
  configured 80% gate.
- `uv build`: exact `yase-0.63.0-py3-none-any.whl` and
  `yase-0.63.0.tar.gz` artifacts build successfully.
- A fresh Python 3.9 environment installs the exact wheel with NumPy/Pillow,
  imports the public API, and round-trips a versioned semantic result.
- `git diff --check` and repository status are clean after validation.

## Included platform work

The release contains robust image, batch, pipeline, video, and realtime
contracts; ONNX I/O binding; OpenVINO asynchronous queues; TensorRT context
pools; Qdrant named vectors and payload indexes; OpenTelemetry spans at facade
and scheduler-stage level; registry-controlled JSON/TOML configuration;
versioned result serialization; and bounded ASGI single/batch extraction.

## Runtime boundary

This CPU-only development host does not have live ONNX Runtime, OpenVINO,
TensorRT/CUDA, Qdrant, Transformers, PaddleOCR, or OpenTelemetry providers
installed. Those adapters are tested through deterministic fakes and lazy
imports, but production CI must add provider-specific jobs, driver versions,
GPU memory profiles, and real model artifacts.

Yase never downloads model weights. Deployment owners remain responsible for
artifact verification, service authentication/rate limiting, model licenses,
and choosing a TensorRT pool size compatible with available device memory.

The remaining roadmap is explicit in the [full package audit](full-package-audit.md):
live provider CI, checkpoint migrations, complete trace propagation through
all sinks, richer declarative deployment config, and keyframe/VLM routing.
