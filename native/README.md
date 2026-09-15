# Optional native acceleration

Yase remains fully functional in Python-only installations. On Linux and
other systems with a C++17 compiler, build the optional extension in place:

```bash
uv run python tools/build_native.py
uv run python -c "from yase.native import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)"
```

The extension currently accelerates pairwise IoU matrices used by tracking.
`yase.native.iou_matrix()` always has a deterministic Python fallback, so the
public API and wheel remain portable when no compiler is available.
