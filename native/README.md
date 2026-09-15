# Optional native acceleration

Yase remains fully functional in Python-only installations. On Linux and
other systems with a C++17 compiler, build the optional extension in place:

```bash
uv run python tools/build_native.py
uv run python -c "from yase.native import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)"
```

The extension accelerates pairwise IoU matrices used by tracking and greedy
non-maximum suppression used by detector post-processing. `iou_matrix()` and
`nms_indices()` always have deterministic Python fallbacks, so the public API
and generic Python wheel remain portable when no compiler is available. The
compiled artifact is intentionally not copied into a `py3-none-any` wheel:
native wheels must be built with platform- and Python-ABI-specific tooling.

The builder needs the matching Python development headers (`Python.h`). On
Ubuntu install `python3-dev` or the exact interpreter package such as
`python3.12-dev`. For non-system interpreters, set
`YASE_PYTHON_INCLUDE_DIR` to the directory containing `Python.h`.
