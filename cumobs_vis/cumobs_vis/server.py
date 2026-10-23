"""Bokeh-serve entry point.

Run with::

    bokeh serve --show cumobs_vis/server.py --args path/to/runs.json
    bokeh serve --show cumobs_vis/server.py --args path/to/a/log_dir   # directory ok

or, more conveniently, via the launcher::

    python launch.py --manifest path/to/runs.json
    python launch.py --manifest path/to/a/log_dir

The source is resolved from (in order): the ``--args`` value, the
``CUMOBS_VIS_MANIFEST`` environment variable, or a ``runs.json`` in the current
directory. It may be a ``runs.json`` **file** or a **directory** of logged runs
(auto-discovered; a ``ProbeSetCollection`` file is split into one run per probe
set). With none of those, the app starts empty (drop a ``.jsonl`` to load).
"""

from __future__ import annotations

import os
import sys

from bokeh.plotting import curdoc

# make the project dir importable so `import cumobs_vis` works when Bokeh serves
# this file by path.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_PKG_DIR)
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

from cumobs_vis.app import build_document  # noqa: E402


def _resolve_manifest() -> str | None:
    for arg in sys.argv[1:]:
        if arg and os.path.exists(arg):
            return arg
    env = os.environ.get("CUMOBS_VIS_MANIFEST")
    if env and os.path.exists(env):
        return env
    if os.path.exists("runs.json"):
        return "runs.json"
    return None


build_document(curdoc(), _resolve_manifest())
