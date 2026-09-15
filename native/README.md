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
and wheel remain portable when no compiler is available.
