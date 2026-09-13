# Third-party notices

The public Yase core (src/yase/core.py, src/yase/video.py, and
src/yase/backends.py) is distributed under the MIT License in LICENSE.

Historical development snapshots contained code adapted from Niantic's
Monodepth2 project. That code is governed by Niantic's non-commercial terms,
is not covered by Yase's MIT grant, and is explicitly excluded from Yase source
and wheel distributions. The supported public API uses independent, locally
supplied backends such as TorchScriptExtractor and never downloads model
weights implicitly.
