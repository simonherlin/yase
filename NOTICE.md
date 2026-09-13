# Third-party notices

The public Yase core (src/yase/core.py, src/yase/video.py, and
src/yase/backends.py) is distributed under the MIT License in LICENSE.

The historical Monodepth2 implementation under src/yase/models/ includes
code adapted from Niantic's Monodepth2 project. That code is governed by its
upstream non-commercial license and is not covered by Yase's MIT grant. It is
kept in the repository for provenance and migration reference only; the
supported public API uses explicit, locally supplied backends such as
TorchScriptExtractor. Applications must review the upstream terms before
enabling or redistributing that historical code.
