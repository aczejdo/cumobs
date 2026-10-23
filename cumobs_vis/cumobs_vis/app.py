"""The Bokeh application — wires every module into the interactive dashboard.

Tech-stack note (deviation from the plan's default): the plan recommends a
Vite + React + uPlot SPA, with a **Bokeh** app as the explicit fallback "if the
repo already has a Python-frontend toolchain / the user prefers Python." This
environment has no Node toolchain and the whole project is Python (the training
side is ``cumobs`` + torch), so the Bokeh fallback is used. Bokeh gives native
linked crosshairs, ``Span`` markers / validity shading, ``varea`` bands, and
server-side file access for the §4i spectrum recompute — everything the plan asks
for, in one ``pip install`` and one ``bokeh serve`` command.

The visualizer itself needs only ``bokeh`` and ``numpy`` — **not** torch or
``cumobs`` — so it stays a lightweight analysis tool.
"""

from __future__ import annotations

import base64
import math
import os
from collections import OrderedDict

import numpy as np
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    DataTable,
    Div,
    FileInput,
    NumericInput,
    RadioGroup,
    Select,
    Slider,
    TableColumn,
    TextInput,
    Toggle,
)

from . import catalog
from .band_provider import DEFAULT_PROVIDER
from .collapse_registry import PanelUnit, resolve_panels
from .config_guard import check_configs
from .data_loader import RunData, load_run_from_text
from .marker_store import CATEGORIES, MarkerStore
from .panels import (
    add_band,
    add_line_trace,
    add_marker,
    add_validity_span,
    build_readout_callback,
    make_reference_figure,
    make_scalar_figure,
    new_shared_crosshair,
)
from .run_store import RunStore
from .validity_spans import compute_spans

# panel primary-key -> per-group base observable (reverse of catalog.GROUP_SERIES)
_PANEL_TO_GROUP_BASE = {v: k for k, v in catalog.GROUP_SERIES.items()}
_RECOMPUTABLE = ("trace_distance", "spectral_drift", "classical_fidelity")

# reference-panel metric-key resolution (real logs use train/loss, accuracy, ...)
_LOSS_KEYS = ["loss", "train/loss", "train_loss", "training/loss", "train/loss/mean"]
_ACC_KEYS = ["acc", "accuracy", "train/accuracy", "train/acc", "train_acc",
             "val/accuracy", "test/accuracy"]

_NAN = float("nan")


def _pick_metric(metrics: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        vals = metrics.get(c)
        if vals is not None and any(_finite(v) for v in vals):
            return c
    return None


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# --- pure transforms -------------------------------------------------------


def _invert(values: list[float]) -> list[float]:
    return [(1.0 - v) if _finite(v) else _NAN for v in values]


def _invert_band(lo, hi):
    # 1 - value flips and swaps the bounds.
    return [(1.0 - h) if _finite(h) else _NAN for h in hi], \
           [(1.0 - l) if _finite(l) else _NAN for l in lo]


def _zscore(values: list[float]):
    finite = [v for v in values if _finite(v)]
    if len(finite) < 2:
        return values, 0.0, 1.0
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    std = math.sqrt(var)
    if std == 0:
        return values, mean, 1.0
    return [((v - mean) / std) if _finite(v) else _NAN for v in values], mean, std


def _classical_distances(ref, cur) -> dict[str, float]:
    """trace_distance / spectral_drift / classical_fidelity between two spectra."""
    r = np.clip(np.asarray(ref, dtype=float), 0.0, None)
    c = np.clip(np.asarray(cur, dtype=float), 0.0, None)
    n = max(r.size, c.size)
    r = np.pad(r, (0, n - r.size))
    c = np.pad(c, (0, n - c.size))
    out: dict[str, float] = {}
    sr, sc = r.sum(), c.sum()
    if sr > 0 and sc > 0:
        p, q = r / sr, c / sc
        out["classical_fidelity"] = float(np.square(np.sqrt(p * q).sum()))
        out["trace_distance"] = 0.5 * float(np.abs(p - q).sum())
    denom = max(float(np.linalg.norm(r)), float(np.linalg.norm(c)))
    if denom > 0:
        out["spectral_drift"] = float(np.linalg.norm(r - c) / denom)
    return out


class VisApp:
    def __init__(self, store: RunStore, manifest_dir: str | None = None) -> None:
        self.store = store
        self.markers = MarkerStore()
        self.band_provider = DEFAULT_PROVIDER
        self.manifest_dir = manifest_dir or os.getcwd()

        # view state
        self.log_global = False
        self.normalize = False
        self.band_k = 1.0
        self.show_bands = True
        self.show_validity = True
        self.include_imag = False
        self.drag_mode = "box"
        self.panel_log: dict[str, bool] = {}
        self.group_mode: dict[str, str] = {}      # gid -> "collapsed" | "overlay"
        self.recomputed: dict = {}                # per-key {steps, values, run, ref_step}
        self.guard_dismissed_sig: tuple = ()

        self._palette_i = len(self.store.runs)
        self._group_index: list[tuple[str, str, str]] = []
        self._selected_marker: int | None = None

        self._init_shared_x()
        self._build_sidebar()

        self.main_col = column(sizing_mode="stretch_width")
        self.root = row(self.sidebar, self.main_col, sizing_mode="stretch_width")

        self.refresh_run_widgets()
        self.refresh_markers()
        self.rebuild_panels()

    # -- domain ------------------------------------------------------------

    def _init_shared_x(self) -> None:
        from bokeh.models import Range1d

        steps: list[int] = []
        for r in self.store.runs:
            steps += r.steps
            steps += r.metrics_steps
        if steps:
            lo, hi = min(steps), max(steps)
            pad = max(1.0, (hi - lo) * 0.02)
            self.shared_x = Range1d(start=lo - pad, end=hi + pad)
        else:
            self.shared_x = Range1d(start=0, end=1)

    # -- sidebar -----------------------------------------------------------

    def _build_sidebar(self) -> None:
        W = 300
        self.title_div = Div(
            text="<h3 style='margin:0'>cumobs · scalar visualizer</h3>"
            "<div style='opacity:.6;font-size:11px'>stacked scalar observables · "
            "locked shared step axis</div>",
            width=W,
        )

        # --- runs ---------------------------------------------------------
        self.run_checkbox = CheckboxGroup(labels=[], active=[], width=W)
        self.run_checkbox.on_change("active", self._on_run_vis)
        self.add_run_input = FileInput(accept=".jsonl,.json", width=W,
                                       title="＋ add run (drop a .jsonl)")
        self.add_run_input.on_change("value", self._on_add_run_file)

        # --- per-group sub-series ----------------------------------------
        self.group_checkbox = CheckboxGroup(labels=[], active=[], width=W)
        self.group_checkbox.on_change("active", self._on_group_vis)

        # --- view toggles -------------------------------------------------
        self.log_toggle = Toggle(label="log-y (all panels)", width=W, height=30)
        self.log_toggle.on_click(self._on_log_global)
        self.norm_toggle = Toggle(label="normalize (z-score per trace)", width=W, height=30)
        self.norm_toggle.on_click(self._on_normalize)

        self.bands_toggle = Toggle(label="show SE bands", active=True, width=145, height=30)
        self.bands_toggle.on_click(self._on_show_bands)
        self.band_k = 1.0
        self.k_slider = Slider(start=0.0, end=3.0, step=0.5, value=1.0,
                               title="band k (± k·se)", width=W)
        self.k_slider.on_change("value_throttled", self._on_band_k)

        self.validity_toggle = Toggle(label="validity shading", active=True,
                                      width=145, height=30)
        self.validity_toggle.on_click(self._on_validity)
        self.imag_toggle = Toggle(label="+ imag-ratio", width=145, height=30)
        self.imag_toggle.on_click(self._on_imag)

        self.drag_radio = RadioGroup(labels=["drag = box-zoom", "drag = pan"],
                                     active=0, width=W)
        self.drag_radio.on_change("active", self._on_drag)
        self.reset_btn = Button(label="reset zoom (full domain)", width=W, height=30)
        self.reset_btn.on_click(self._on_reset_zoom)

        # --- markers ------------------------------------------------------
        self.marker_label = TextInput(title="marker label", width=W)
        self.marker_step = NumericInput(title="step", mode="float", width=140)
        self.marker_cat = Select(title="category", options=list(CATEGORIES),
                                 value="manual", width=140)
        self.marker_add_btn = Button(label="add", button_type="primary", width=90, height=30)
        self.marker_add_btn.on_click(self._on_add_marker)
        self.marker_upd_btn = Button(label="update sel.", width=100, height=30)
        self.marker_upd_btn.on_click(self._on_update_marker)
        self.marker_del_btn = Button(label="delete sel.", width=100, height=30)
        self.marker_del_btn.on_click(self._on_delete_marker)

        self.marker_source = ColumnDataSource(dict(id=[], label=[], step=[], category=[]))
        self.marker_source.selected.on_change("indices", self._on_marker_select)
        self.marker_table = DataTable(
            source=self.marker_source, width=W, height=140,
            columns=[
                TableColumn(field="step", title="step", width=60),
                TableColumn(field="label", title="label"),
                TableColumn(field="category", title="cat", width=80),
            ],
        )
        self.marker_export_btn = Button(label="export markers.json", width=145, height=30)
        self.marker_export_btn.on_click(self._on_export_markers)
        self.marker_import_input = FileInput(accept=".json", width=145,
                                             title="import markers")
        self.marker_import_input.on_change("value", self._on_import_markers)
        self.marker_status = Div(text="", width=W, styles={"font-size": "11px",
                                                           "opacity": "0.7"})

        # --- click-to-set-reference (§4i) --------------------------------
        self.recompute_status = Div(text="", width=W, styles={"font-size": "11px"})
        self.clear_recompute_btn = Button(label="clear recomputed", width=W, height=28)
        self.clear_recompute_btn.on_click(self._on_clear_recomputed)

        def _sec(title: str) -> Div:
            return Div(text=f"<b style='font-size:12px'>{title}</b>",
                       styles={"margin-top": "8px", "border-top": "1px solid #e2e8f0",
                               "padding-top": "6px"}, width=W)

        self.sidebar = column(
            self.title_div,
            _sec("Runs (color = run, everywhere)"),
            self.run_checkbox,
            self.add_run_input,
            _sec("Per-group sub-series"),
            self.group_checkbox,
            _sec("Scales"),
            self.log_toggle,
            self.norm_toggle,
            row(self.bands_toggle),
            self.k_slider,
            row(self.validity_toggle, self.imag_toggle),
            _sec("Navigation"),
            self.drag_radio,
            self.reset_btn,
            _sec("Event markers"),
            self.marker_label,
            row(self.marker_step, self.marker_cat),
            row(self.marker_add_btn, self.marker_upd_btn, self.marker_del_btn),
            self.marker_table,
            row(self.marker_export_btn, self.marker_import_input),
            self.marker_status,
            _sec("Click-to-set-reference (§4i)"),
            self.recompute_status,
            self.clear_recompute_btn,
            width=320,
        )

    # -- dynamic sidebar refreshers ---------------------------------------

    def refresh_run_widgets(self) -> None:
        runs = self.store.runs
        self.run_checkbox.labels = [r.name for r in runs]
        self.run_checkbox.active = [i for i, r in enumerate(runs)
                                    if self.store.is_visible(r.name)]

        # group sub-series list across all runs
        self._group_index = []
        labels = []
        for r in runs:
            for base in self.store.group_bases(r):
                for lab in r.group_labels(base):
                    self._group_index.append((r.name, base, lab))
                    labels.append(f"{r.name} · {base}/{lab}")
        self.group_checkbox.labels = labels
        self.group_checkbox.active = [
            i for i, (rn, base, lab) in enumerate(self._group_index)
            if self.store.is_group_visible(rn, base, lab)
        ]

        # tap / recompute status
        ref_run = self._recompute_run()
        if ref_run is not None:
            self.recompute_status.text = (
                f"<span style='color:#166534'>enabled</span> — click any panel to "
                f"recompute classical distances (trace_distance / spectral_drift / "
                f"classical_fidelity) against that step for run "
                f"<b>{ref_run.name}</b> (covariance_spectrum is logged).")
        else:
            self.recompute_status.text = (
                "<span style='color:#991b1b'>disabled</span> — no run logs "
                "<code>covariance_spectrum</code> (a full-vector <code>.npz</code> "
                "sidecar). Recomputing classical distances needs the per-step "
                "spectra. Quantum/CKA distances are never recomputed client-side.")

    def refresh_markers(self) -> None:
        ms = self.markers.all()
        self.marker_source.data = dict(
            id=[m.id for m in ms],
            label=[m.label for m in ms],
            step=[m.step for m in ms],
            category=[m.category for m in ms],
        )

    # -- main rebuild ------------------------------------------------------

    def rebuild_panels(self) -> None:
        crosshair = new_shared_crosshair()
        figs = []
        readout_sources: list = []
        readout_labels: list[str] = []
        readout_colors: list[str] = []

        # readout Div (crosshairController)
        readout = Div(
            text="<div style='opacity:.6'>hover a panel…</div>",
            width=320,
            styles={"border": "1px solid #e2e8f0", "border-radius": "6px",
                    "padding": "6px 8px", "background": "#ffffff",
                    "max-height": "220px", "overflow": "auto", "font-size": "12px"},
        )

        # --- config-guard banner (§4d) -----------------------------------
        warnings = check_configs(self.store.visible_runs())
        sig = tuple(w.field for w in warnings)
        banner_children = []
        if warnings and sig != self.guard_dismissed_sig:
            items = "".join(f"<li>{w.render()}</li>" for w in warnings)
            dismiss = Button(label="dismiss", width=80, height=26)

            def _dismiss(_s=sig):
                self.guard_dismissed_sig = _s
                self.rebuild_panels()

            dismiss.on_click(lambda: _dismiss())
            banner = Div(
                text=f"<b>⚠ config-compatibility warning</b><ul style='margin:4px 0'>"
                     f"{items}</ul>",
                sizing_mode="stretch_width",
                styles={"background": "#fef3c7", "border": "1px solid #f59e0b",
                        "border-radius": "6px", "padding": "8px 12px",
                        "color": "#78350f", "font-size": "12px"},
            )
            banner_children = [row(banner, dismiss, sizing_mode="stretch_width")]

        # --- reference panel ---------------------------------------------
        ref_fig = self._build_reference(crosshair, readout_sources,
                                        readout_labels, readout_colors)
        figs.append(ref_fig)

        # --- scalar panels ------------------------------------------------
        present = self.store.present_scalar_keys(only_visible=True)
        sections = resolve_panels(present)
        panel_children = []
        for fam_id, units in sections.items():
            panel_children.append(Div(
                text=f"<div style='font-weight:700;color:#334155;font-size:12px;"
                     f"margin-top:10px'>{catalog.FAMILY_TITLES.get(fam_id, fam_id)}"
                     f"</div>",
                sizing_mode="stretch_width",
            ))
            for unit in units:
                col, fig = self._build_panel(
                    unit, crosshair, readout_sources, readout_labels, readout_colors)
                figs.append(fig)
                panel_children.append(col)

        # --- crosshairController: shared readout on every figure ---------
        cb = build_readout_callback(readout, readout_sources, readout_labels,
                                    readout_colors)
        for f in figs:
            f.js_on_event("mousemove", cb)

        legend = self._build_legend()
        self.main_col.children = (
            banner_children
            + [row(legend, readout, sizing_mode="stretch_width"), ref_fig]
            + panel_children
        )

    # -- reference panel ---------------------------------------------------

    def _build_reference(self, crosshair, r_srcs, r_labels, r_colors):
        runs = self.store.visible_runs()
        loss_key = {r.name: _pick_metric(r.metrics, _LOSS_KEYS)
                    for r in runs if r.has_metrics()}
        acc_key = {r.name: _pick_metric(r.metrics, _ACC_KEYS)
                   for r in runs if r.has_metrics()}
        any_metrics = any(loss_key.get(r.name) for r in runs)
        has_acc = any(acc_key.get(r.name) for r in runs)
        fig = make_reference_figure(self.shared_x, crosshair, has_acc=has_acc,
                                    drag_mode=self.drag_mode)

        if any_metrics:
            seen_metrics: set[int] = set()   # dedupe shared (global) metrics
            for r in runs:
                lk = loss_key.get(r.name)
                if not lk:
                    continue
                sig = id(r.metrics)
                if sig in seen_metrics:
                    continue
                seen_metrics.add(sig)
                # shared training metrics are global -> label without run name.
                prefix = "" if len(seen_metrics) == 1 and len(runs) > 1 else f"{r.name} · "
                src = add_line_trace(fig, r.metrics_steps, r.metrics[lk],
                                     color=r.color, label=f"{prefix}{lk}")
                r_srcs.append(src); r_labels.append(f"{prefix}{lk}"); r_colors.append(r.color)
                ak = acc_key.get(r.name)
                if has_acc and ak:
                    albl = f"{prefix}{ak}"
                    s2 = add_line_trace(fig, r.metrics_steps, r.metrics[ak],
                                        color=r.color, label=albl,
                                        dash="dashed", width=1.4, y_range_name="acc")
                    r_srcs.append(s2); r_labels.append(albl); r_colors.append(r.color)
        else:
            # fallback: total_first_cumulant (mean probe loss) per run (plan §1)
            fig.yaxis.axis_label = "total_first_cumulant"
            fig.title.text = ("Reference — total_first_cumulant fallback "
                              "(no metrics file supplied)")
            for r in runs:
                vals = r.series.get("total_first_cumulant")
                if vals is None:
                    continue
                lbl = f"{r.name} · total_first_cumulant"
                src = add_line_trace(fig, r.steps, vals, color=r.color, label=lbl)
                r_srcs.append(src); r_labels.append(lbl); r_colors.append(r.color)

        # markers + validity on the reference too (shared step axis)
        self._decorate_spans_markers(fig)
        return fig

    # -- one scalar panel --------------------------------------------------

    def _build_panel(self, unit: PanelUnit, crosshair, r_srcs, r_labels, r_colors):
        panel_log = self.panel_log.get(unit.panel_id, self.log_global)
        use_log = panel_log and not self.normalize
        fig = make_scalar_figure(self.shared_x, crosshair, log=use_log,
                                 drag_mode=self.drag_mode, tap=True)
        fig.on_event(Tap, self._on_tap)

        # which member keys to draw
        if unit.is_group:
            mode = self.group_mode.get(unit.panel_id, "collapsed")
            if mode == "overlay":
                members = list(unit.members)
            else:
                members = [m for m in unit.members if m.key == unit.representative] \
                          or [unit.members[0]]
        else:
            members = list(unit.members)

        panel_key = unit.representative
        group_base = _PANEL_TO_GROUP_BASE.get(panel_key)

        for run in self.store.visible_runs():
            for mi, member in enumerate(members):
                self._draw_member(fig, run, member, mi, len(members),
                                  r_srcs, r_labels, r_colors)
            # nested per-group sub-series on the matching panel (§1, §4b)
            if group_base and group_base in run.groups:
                for gi, lab in enumerate(run.group_labels(group_base)):
                    if not self.store.is_group_visible(run.name, group_base, lab):
                        continue
                    self._draw_group_series(fig, run, group_base, lab, gi,
                                            r_srcs, r_labels, r_colors)

        # validity shading + markers
        self._decorate_spans_markers(fig)

        # header row: title (+tooltip), per-panel log toggle, group controls
        header = self._panel_header(unit)
        return column(header, fig, sizing_mode="stretch_width"), fig

    def _draw_member(self, fig, run: RunData, member, idx, n_members,
                     r_srcs, r_labels, r_colors) -> None:
        key = member.key
        vals = run.series.get(key)
        if vals is None or not any(_finite(v) for v in vals):
            return
        xs = run.steps
        ys = list(vals)
        band = None
        if self.show_bands and not self.normalize:
            b = self.band_provider.band(run, key, self.band_k)
            if b is not None:
                band = b
        if member.invert:
            ys = _invert(ys)
            if band is not None:
                band = _invert_band(band[0], band[1])
        if self.normalize:
            ys, mean, std = _zscore(ys)
        dash = "solid" if n_members == 1 else \
            ["solid", "dashed", "dotted", "dashdot"][idx % 4]
        label = run.name if n_members == 1 else f"{run.name} · {member.label}"

        if band is not None:
            add_band(fig, xs, band[0], band[1], color=run.color, alpha=0.16)
        src = add_line_trace(fig, xs, ys, color=run.color, label=label, dash=dash)
        r_srcs.append(src); r_labels.append(label); r_colors.append(run.color)

        # recomputed overlay from click-to-set-reference (§4i)
        rec = self.recomputed.get(key)
        if rec and rec.get("run") == run.name:
            r_ys = list(rec["values"])
            if member.invert:
                r_ys = _invert(r_ys)
            if self.normalize:
                r_ys, _, _ = _zscore(r_ys)
            lbl = f"{run.name} · {key} vs step {rec['ref_step']} (recomputed)"
            s2 = add_line_trace(fig, rec["steps"], r_ys, color=run.color,
                                label=lbl, dash="dashed", width=1.4, alpha=0.7)
            r_srcs.append(s2); r_labels.append(lbl); r_colors.append(run.color)

    def _draw_group_series(self, fig, run: RunData, base, lab, gi,
                           r_srcs, r_labels, r_colors) -> None:
        vals = run.groups.get(base, {}).get(lab)
        if vals is None or not any(_finite(v) for v in vals):
            return
        ys = list(vals)
        band = None
        if self.show_bands and not self.normalize:
            band = self.band_provider.band_group(run, base, lab, self.band_k)
        if self.normalize:
            ys, _, _ = _zscore(ys)
        label = f"{run.name} · {base}/{lab}"
        if band is not None:
            add_band(fig, run.steps, band[0], band[1], color=run.color, alpha=0.10)
        src = add_line_trace(fig, run.steps, ys, color=run.color, label=label,
                             dash="dashed", width=1.3, alpha=0.8)
        r_srcs.append(src); r_labels.append(label); r_colors.append(run.color)

    def _decorate_spans_markers(self, fig) -> None:
        if self.show_validity:
            for run in self.store.visible_runs():
                for span in compute_spans(run, include_imag_ratio=self.include_imag):
                    add_validity_span(fig, span)
        for m in self.markers.all():
            add_marker(fig, m)

    def _panel_header(self, unit: PanelUnit):
        from bokeh.models import Div as _Div
        tip = catalog.tooltip_for(unit.representative)
        note = f" <span style='opacity:.6'>· {unit.note}</span>" if unit.note else ""
        title = _Div(
            text=f"<span title=\"{tip}\" style='font-weight:600;cursor:help'>"
                 f"{unit.title}</span>{note}",
            sizing_mode="stretch_width",
        )
        log_cb = CheckboxGroup(
            labels=["log"], width=52, height=24,
            active=[0] if self.panel_log.get(unit.panel_id, self.log_global) else [],
        )
        log_cb.on_change("active",
                         lambda a, o, n, pid=unit.panel_id: self._on_panel_log(pid, n))
        children = [title, log_cb]
        if unit.is_group:
            mode = self.group_mode.get(unit.panel_id, "collapsed")
            grp = RadioGroup(labels=["collapsed", "overlay"],
                             active=0 if mode == "collapsed" else 1,
                             width=150, height=24, inline=True)
            grp.on_change("active",
                          lambda a, o, n, gid=unit.panel_id: self._on_group_mode(gid, n))
            children.append(grp)
        return row(*children, sizing_mode="stretch_width")

    def _build_legend(self):
        chips = []
        for r in self.store.visible_runs():
            chips.append(
                f"<span style='margin-right:10px;white-space:nowrap'>"
                f"<span style='color:{r.color};font-size:14px'>&#9632;</span> "
                f"{r.name}</span>")
        text = "".join(chips) or "<span style='opacity:.6'>no visible runs</span>"
        return Div(text=f"<div style='font-size:12px'>{text}</div>",
                   sizing_mode="stretch_width")

    # -- recompute run selection ------------------------------------------

    def _recompute_run(self) -> RunData | None:
        for r in self.store.visible_runs():
            if r.has_covariance_spectrum:
                return r
        return None

    # ===================================================================
    # callbacks
    # ===================================================================

    def _on_run_vis(self, attr, old, new) -> None:
        active = set(new)
        for i, r in enumerate(self.store.runs):
            self.store.set_visible(r.name, i in active)
        self.refresh_run_widgets()
        self.rebuild_panels()

    def _on_group_vis(self, attr, old, new) -> None:
        active = set(new)
        for i, (rn, base, lab) in enumerate(self._group_index):
            self.store.set_group_visible(rn, base, lab, i in active)
        self.rebuild_panels()

    def _on_log_global(self, active: bool) -> None:
        self.log_global = active
        self.panel_log.clear()  # global override resets per-panel choices
        self.rebuild_panels()

    def _on_panel_log(self, panel_id: str, new_active) -> None:
        self.panel_log[panel_id] = bool(new_active)
        self.rebuild_panels()

    def _on_group_mode(self, gid: str, new_active) -> None:
        self.group_mode[gid] = "overlay" if new_active == 1 else "collapsed"
        self.rebuild_panels()

    def _on_normalize(self, active: bool) -> None:
        self.normalize = active
        self.rebuild_panels()

    def _on_show_bands(self, active: bool) -> None:
        self.show_bands = active
        self.rebuild_panels()

    def _on_band_k(self, attr, old, new) -> None:
        self.band_k = float(new)
        self.rebuild_panels()

    def _on_validity(self, active: bool) -> None:
        self.show_validity = active
        self.rebuild_panels()

    def _on_imag(self, active: bool) -> None:
        self.include_imag = active
        self.rebuild_panels()

    def _on_drag(self, attr, old, new) -> None:
        self.drag_mode = "box" if new == 0 else "pan"
        self.rebuild_panels()

    def _on_reset_zoom(self) -> None:
        steps: list[int] = []
        for r in self.store.runs:
            steps += r.steps + r.metrics_steps
        if steps:
            lo, hi = min(steps), max(steps)
            pad = max(1.0, (hi - lo) * 0.02)
            self.shared_x.start, self.shared_x.end = lo - pad, hi + pad

    # --- markers ----------------------------------------------------------

    def _on_add_marker(self) -> None:
        step = self.marker_step.value
        if step is None:
            self.marker_status.text = "enter a step first."
            return
        self.markers.add(self.marker_label.value or "", step, self.marker_cat.value)
        self.refresh_markers()
        self.rebuild_panels()

    def _on_update_marker(self) -> None:
        if self._selected_marker is None:
            self.marker_status.text = "select a marker row to update."
            return
        self.markers.update(self._selected_marker,
                            label=self.marker_label.value,
                            step=self.marker_step.value,
                            category=self.marker_cat.value)
        self.refresh_markers()
        self.rebuild_panels()

    def _on_delete_marker(self) -> None:
        if self._selected_marker is None:
            self.marker_status.text = "select a marker row to delete."
            return
        self.markers.delete(self._selected_marker)
        self._selected_marker = None
        self.refresh_markers()
        self.rebuild_panels()

    def _on_marker_select(self, attr, old, new) -> None:
        if not new:
            self._selected_marker = None
            return
        i = new[0]
        data = self.marker_source.data
        self._selected_marker = data["id"][i]
        self.marker_label.value = str(data["label"][i])
        self.marker_step.value = data["step"][i]
        self.marker_cat.value = str(data["category"][i])

    def _on_export_markers(self) -> None:
        path = os.path.join(self.manifest_dir, "markers.json")
        self.markers.export_file(path)
        self.marker_status.text = f"exported {len(self.markers)} markers → {path}"

    def _on_import_markers(self, attr, old, new) -> None:
        if not new:
            return
        try:
            text = base64.b64decode(new).decode("utf-8")
            count = self.markers.load_json(text, replace=True)
        except Exception as exc:  # tolerate a bad file
            self.marker_status.text = f"import failed: {exc}"
            return
        self.refresh_markers()
        self.rebuild_panels()
        self.marker_status.text = f"imported {count} markers."

    # --- add run via drop -------------------------------------------------

    def _on_add_run_file(self, attr, old, new) -> None:
        if not new:
            return
        try:
            text = base64.b64decode(new).decode("utf-8")
        except Exception:
            return
        from .data_loader import DEFAULT_PALETTE
        name = (self.add_run_input.filename or "dropped").rsplit(".", 1)[0]
        color = DEFAULT_PALETTE[self._palette_i % len(DEFAULT_PALETTE)]
        self._palette_i += 1
        run = load_run_from_text(text, name=name, color=color)
        self.store.add_run(run)
        self._init_shared_x()  # extend the domain to the new run
        self._on_reset_zoom()
        self.refresh_run_widgets()
        self.rebuild_panels()

    # --- click-to-set-reference (§4i) ------------------------------------

    def _on_tap(self, event) -> None:
        run = self._recompute_run()
        if run is None or event.x is None:
            return
        spectra = run.load_covariance_spectra()
        if not spectra:
            return
        steps = sorted(spectra.keys())
        ref_step = min(steps, key=lambda s: abs(s - event.x))
        ref = spectra[ref_step]
        acc: dict[str, list[float]] = {k: [] for k in _RECOMPUTABLE}
        for s in steps:
            d = _classical_distances(ref, spectra[s])
            for k in _RECOMPUTABLE:
                acc[k].append(d.get(k, _NAN))
        for k in _RECOMPUTABLE:
            self.recomputed[k] = {"steps": steps, "values": acc[k],
                                  "run": run.name, "ref_step": ref_step}
        self.recompute_status.text = (
            f"recomputed classical distances for <b>{run.name}</b> vs step "
            f"<b>{ref_step}</b> (dashed overlays).")
        self.rebuild_panels()

    def _on_clear_recomputed(self) -> None:
        self.recomputed.clear()
        self.refresh_run_widgets()
        self.rebuild_panels()


# --- document entry --------------------------------------------------------


def build_document(doc, manifest_path: str | None = None) -> VisApp:
    """Load a manifest **or a directory** (if given) and attach the app.

    ``manifest_path`` may be a ``runs.json`` file or a directory of logged runs
    (auto-discovered). ``ProbeSetCollection`` files are split into one run per
    probe set."""
    from .data_loader import load_source

    runs = []
    manifest_dir = None
    if manifest_path and os.path.exists(manifest_path):
        runs = load_source(manifest_path)
        manifest_dir = (manifest_path if os.path.isdir(manifest_path)
                        else os.path.dirname(os.path.abspath(manifest_path)))

    store = RunStore(runs)
    # keep the first view readable when a collection expands to many runs.
    if len(store.runs) > 3:
        for r in store.runs[2:]:
            store.set_visible(r.name, False)

    app = VisApp(store, manifest_dir=manifest_dir)
    doc.add_root(app.root)
    doc.title = "cumobs · scalar observable visualizer"
    return app
