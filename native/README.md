# Optional native acceleration

Yase remains fully functional in Python-only installations. The extension is
an optimization layer, not a second vision runtime: ONNX Runtime, OpenVINO,
TensorRT, and PyTorch continue to own model inference. The local C++17 module
contains the small CPU kernels that are reused by tracking and detector
post-processing:

- pairwise axis-aligned IoU;
- deterministic class-aware greedy NMS.

Build the optional extension in place during development:

```bash
uv run python tools/build_native.py
uv run python -c "from yase.native import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)"
```

`iou_matrix()` and `nms_indices()` always have deterministic Python fallbacks,
so the public API remains identical without a compiler. The regular portable
wheel intentionally contains no binary. To build a platform wheel with the
extension included:

```bash
YASE_BUILD_NATIVE=1 uv run python -m build --wheel
# or
make build-native
```

The resulting wheel is platform- and Python-ABI-specific. For publication,
build it with cibuildwheel on the supported OS/Python matrix rather than
copying a local `.so` between machines.

The builder needs the matching Python development headers (`Python.h`). On
Ubuntu install `python3-dev` or the exact interpreter package such as
`python3.12-dev`. For non-system interpreters, set
`YASE_PYTHON_INCLUDE_DIR` to the directory containing `Python.h`.
