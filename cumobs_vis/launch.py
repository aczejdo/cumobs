"""Convenience launcher for the cumobs scalar visualizer.

Wraps ``bokeh serve`` so you don't have to remember the incantation::

    python launch.py --manifest runs.json          # load a manifest and open a tab
    python launch.py --manifest runs.json --port 5010
    python launch.py                                # start empty (drag-drop a .jsonl)

It shells out to ``python -m bokeh serve`` using the *current* interpreter, so it
picks up the same virtual environment. Ctrl-C stops the server.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch the cumobs scalar visualizer.")
    ap.add_argument("--manifest", default=None,
                    help="path to a runs.json manifest (optional)")
    ap.add_argument("--port", type=int, default=5006)
    ap.add_argument("--no-show", action="store_true",
                    help="do not auto-open a browser tab")
    args = ap.parse_args()

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    server = os.path.join(proj_dir, "cumobs_vis", "server.py")

    cmd = [sys.executable, "-m", "bokeh", "serve", server, "--port", str(args.port)]
    if not args.no_show:
        cmd.append("--show")

    env = dict(os.environ)
    if args.manifest:
        manifest = os.path.abspath(args.manifest)
        if not os.path.exists(manifest):
            print(f"manifest not found: {manifest}", file=sys.stderr)
            return 2
        cmd += ["--args", manifest]
        env["CUMOBS_VIS_MANIFEST"] = manifest

    print("launching:", " ".join(cmd))
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
