"""cumobs_vis — a local, interactive dashboard for the *scalar* observables
produced by the ``cumobs`` monitor.

This package is a scoped analysis tool (see
``../scalar_observable_visualizer_plan.md``). It reads the flattened JSONL rows
written by :class:`cumobs.JSONLLogger`, plus optional metrics / config sidecars,
and renders stacked scalar line panels with a locked shared step axis, a pinned
loss/acc reference panel, uncertainty bands, validity shading, a config
compatibility guard, and event markers.

Vector/matrix observables (spectra, heatmaps, per-mode ribbons) are explicitly
out of scope.

The modules mirror the plan's architecture:

* :mod:`cumobs_vis.catalog`          — the scalar key catalog (families, tooltips)
* :mod:`cumobs_vis.collapse_registry`— the redundancy / collapse groups
* :mod:`cumobs_vis.data_loader`      — manifest + JSONL + metrics ingestion
* :mod:`cumobs_vis.run_store`        — loaded runs, visibility, colors
* :mod:`cumobs_vis.band_provider`    — pluggable uncertainty-band interface
* :mod:`cumobs_vis.config_guard`     — pairwise config-compatibility check
* :mod:`cumobs_vis.validity_spans`   — buffer-fill / dead-window shading
* :mod:`cumobs_vis.marker_store`     — event markers (CRUD + JSON persistence)
* :mod:`cumobs_vis.reductions`       — optional per-mode -> scalar reduction hook
* :mod:`cumobs_vis.panels`           — the Bokeh panel grid + crosshair controller
"""

from __future__ import annotations

__version__ = "0.1.0"
