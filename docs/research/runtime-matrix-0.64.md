# Runtime matrix — Yase 0.64.0

Date of the local run: 2026-09-15. This report records executable smoke tests,
not just package-import checks. The models used by the tests are tiny synthetic
mean-reduction graphs, so the result proves the adapter/provider contract and
not the quality of a production model.

## Selected Python policy

Yase uses Python 3.12 as its reference interpreter and supports Python 3.10,
3.11, 3.12, and 3.13. The repository pins the development interpreter through
`.python-version`; it does not replace the operating system's `/usr/bin/python3`.
Python 3.9 is outside the current OpenVINO/RF-DETR intersection, while Python
3.14 is deliberately deferred until the optional-provider matrix proves it.

The policy is based on the current upstream intersections:

- [Python version status](https://devguide.python.org/versions/)
- [OpenVINO system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html)
- [OpenVINO 2025 release notes](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html)
- [RF-DETR documentation](https://github.com/roboflow/rf-detr/blob/develop/docs/index.md)

## Local machine

| Capability | Observed value | Result |
| --- | --- | --- |
| OS | Ubuntu 24.04.5 LTS | supported development host |
| Python | 3.12.3 (`uv` project environment) | pass |
| CPU | Intel Core i7-6820HQ | pass |
| NVIDIA GPU | Quadro M3000M, 4 GiB, compute capability 5.2 | legacy hardware |
| NVIDIA driver | 580.173.02 | detected |
| CUDA toolkit | `nvcc` not installed | CUDA compiler unavailable |

## Executed providers

| Runtime | Version | Provider/device | Smoke result |
| --- | --- | --- | --- |
| OpenVINO | 2026.3.1 | CPU | pass: sync batch and `AsyncInferQueue` |
| OpenVINO | 2026.3.1 | GPU | pass: tiny graph compiled and executed |
| OpenVINO | 2026.3.1 | AUTO | pass: tiny graph compiled and executed |
| ONNX Runtime | 1.30.0 | CPUExecutionProvider | pass: batch and I/O binding |
| ONNX Runtime | 1.30.0 | AzureExecutionProvider available | not selected for local inference |
| OpenTelemetry API | 1.44.0 | no-op default tracer | pass: span creation |
| TensorRT | not installed | CUDA/TensorRT | hardware-gated; not claimed |

OpenVINO reported `CPU` and `GPU` devices, with the GPU device identified as
the local Quadro. This validates the OpenVINO plugin on this machine for the
synthetic graph; production models still require model-specific accuracy and
performance validation.

## TensorRT decision

TensorRT was not installed merely to produce an import result. The current
[TensorRT prerequisites](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html)
target Turing-class GPUs or newer, whereas this machine exposes Maxwell
compute capability 5.2. A current TensorRT wheel would therefore not provide a
valid local execution target. Yase keeps the TensorRT adapter and context pool
available for deployment hosts with a supported GPU, while this host uses
OpenVINO GPU/AUTO or ONNX CPU as its validated paths.

## Reproduction

```bash
uv sync --extra openvino --extra onnx --extra observability
uv run python -c 'import openvino as ov; print(ov.get_version(), ov.Core().available_devices)'
uv run python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
make check
```

The complete Python test suite remains deterministic: 143 tests pass under
Python 3.12.3. The provider smoke commands are intentionally kept separate
from the base test suite because optional runtimes and hardware vary by host.
