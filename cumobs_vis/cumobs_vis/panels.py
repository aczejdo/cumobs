"""panelGrid + referencePanel + crosshairController — the Bokeh rendering layer.

Pure builders: they take already-transformed data (invert / z-score are applied
by the app) and return Bokeh models. The app (:mod:`cumobs_vis.app`) owns state
and wiring; keeping the glyph plumbing here makes it swappable.

Highlights that realise the plan:

* every scalar figure shares one x-range object (locked shared step axis, §3);
* a single :class:`CrosshairTool` instance is shared across all figures, so the
  vertical line tracks across the whole stack (linked crosshair, §4g);
* :func:`build_readout_callback` is the crosshairController's readout: a JS
  ``mousemove`` handler that fills one Div with *every* visible trace's value at
  the hovered step;
* bands are ``varea`` drawn *underneath* the line (§4c); validity regions are
  ``BoxAnnotation`` underlays (§4e); markers are ``Span`` overlays (§4h).
"""

from __future__ import annotations

from bokeh.models import (
    BoxAnnotation,
    BoxZoomTool,
    CrosshairTool,
    CustomJS,
    DataRange1d,
    Div,
    LinearAxis,
    PanTool,
    Range1d,
    ResetTool,
    Span,
    TapTool,
    WheelZoomTool,
)
from bokeh.plotting import figure

from .marker_store import Marker, style_for
from .validity_spans import KIND_STYLES, ValiditySpan

PANEL_HEIGHT = 150
REFERENCE_HEIGHT = 190
DASHES = ["solid", "dashed", "dotted", "dashdot"]

_AXIS_LABEL_FONT = "11px"
_TITLE = {"text_font_size": "12px", "text_font_style": "bold"}


def new_shared_crosshair() -> CrosshairTool:
    """One vertical linked crosshair, shared across every figure."""
    return CrosshairTool(
        dimensions="height",
        line_color="#475569",
        line_alpha=0.7,
        line_width=1,
    )


def _apply_tools(fig, crosshair: CrosshairTool, drag_mode: str, tap: bool) -> None:
    pan = PanTool(dimensions="width")
    box = BoxZoomTool(dimensions="width")
    wheel = WheelZoomTool(dimensions="width")
    tools = [pan, box, wheel, ResetTool()]
    if tap:
        tools.append(TapTool())
    fig.add_tools(*tools)
    fig.add_tools(crosshair)
    fig.toolbar.active_scroll = wheel
    fig.toolbar.active_drag = box if drag_mode == "box" else pan


def make_scalar_figure(shared_x, crosshair, *, log: bool, drag_mode: str,
                       tap: bool = True):
    """A single stacked scalar panel (no title — the app draws a header row)."""
    fig = figure(
        height=PANEL_HEIGHT,
        sizing_mode="stretch_width",
        x_range=shared_x,
        y_axis_type="log" if log else "linear",
        tools="",
        toolbar_location=None,
        outline_line_color="#e2e8f0",
        min_border_left=64,
        min_border_right=16,
    )
    _apply_tools(fig, crosshair, drag_mode, tap)
    fig.xaxis.axis_label = None
    fig.yaxis.axis_label_text_font_size = _AXIS_LABEL_FONT
    fig.grid.grid_line_alpha = 0.25
    fig.background_fill_color = "#fbfcfe"
    return fig


def make_reference_figure(shared_x, crosshair, *, has_acc: bool, drag_mode: str):
    """The pinned top reference panel (loss on the left axis, acc on the right)."""
    fig = figure(
        height=REFERENCE_HEIGHT,
        sizing_mode="stretch_width",
        x_range=shared_x,
        tools="",
        toolbar_location="above",
        outline_line_color="#cbd5e1",
        min_border_left=64,
        min_border_right=64,
        title="Reference — training loss / acc (x-domain driver)",
    )
    fig.title.update(**_TITLE)
    _apply_tools(fig, crosshair, drag_mode, tap=False)
    fig.xaxis.axis_label = "step"
    fig.yaxis.axis_label = "loss"
    if has_acc:
        fig.extra_y_ranges = {"acc": DataRange1d()}
        fig.add_layout(LinearAxis(y_range_name="acc", axis_label="acc"), "right")
    fig.background_fill_color = "#fafafa"
    return fig


def add_line_trace(fig, xs, ys, *, color: str, label: str, dash: str = "solid",
                   width: float = 1.8, alpha: float = 0.95, y_range_name=None):
    """Add a line and return its ColumnDataSource.

    Per-trace tooltips are intentionally omitted — the linked crosshair readout
    (:func:`build_readout_callback`) is the single hover surface, showing *every*
    visible trace's value at once (plan §4g)."""
    data = {"x": list(xs), "y": list(ys)}
    renderer = fig.line(
        "x", "y", source=data, line_color=color, line_width=width,
        line_alpha=alpha, line_dash=dash,
        **({"y_range_name": y_range_name} if y_range_name else {}),
    )
    renderer.name = label
    return renderer.data_source


def add_band(fig, xs, lo, hi, *, color: str, alpha: float = 0.15, y_range_name=None):
    """A varea band drawn *underneath* the line (level='underlay')."""
    r = fig.varea(
        x="x", y1="lo", y2="hi",
        source={"x": list(xs), "lo": list(lo), "hi": list(hi)},
        fill_color=color, fill_alpha=alpha,
        **({"y_range_name": y_range_name} if y_range_name else {}),
    )
    r.level = "underlay"
    return r


def add_validity_span(fig, span: ValiditySpan) -> None:
    color, alpha, hatch = KIND_STYLES.get(span.kind, ("#94a3b8", 0.12, None))
    kwargs = dict(fill_color=color, fill_alpha=alpha, level="underlay",
                  line_alpha=0.0)
    if span.start is not None:
        kwargs["left"] = span.start
    if span.end is not None:
        kwargs["right"] = span.end
    if hatch:
        kwargs.update(hatch_pattern=hatch, hatch_alpha=alpha, hatch_color=color)
    fig.add_layout(BoxAnnotation(**kwargs))


def add_marker(fig, marker: Marker) -> None:
    style = style_for(marker.category)
    fig.add_layout(
        Span(location=marker.step, dimension="height",
             line_color=style["line_color"], line_dash=style["line_dash"],
             line_width=style["line_width"], line_alpha=0.85)
    )


# --- crosshairController: the shared readout ------------------------------

_READOUT_JS = """
function nearestIdx(xs, x) {
  if (!xs || xs.length === 0) return -1;
  const hi = xs.length - 1;
  if (x <= xs[0]) return 0;
  if (x >= xs[hi]) return hi;
  let lo = 0, h = hi;
  while (lo <= h) {
    const mid = (lo + h) >> 1;
    if (xs[mid] === x) return mid;
    if (xs[mid] < x) lo = mid + 1; else h = mid - 1;
  }
  return (Math.abs(xs[lo] - x) < Math.abs(xs[h] - x)) ? lo : h;
}
function fmt(v) {
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(3);
  return (Math.round(v * 1e4) / 1e4).toString();
}
const x = cb_obj.x;
if (x === null || x === undefined) { return; }
let rows = [];
for (let t = 0; t < srcs.length; t++) {
  const xs = srcs[t].data['x'];
  const ys = srcs[t].data['y'];
  const i = nearestIdx(xs, x);
  if (i < 0) continue;
  const yv = ys[i];
  if (yv === null || yv === undefined || Number.isNaN(yv)) continue;
  rows.push(
    "<div style='white-space:nowrap'><span style='color:" + colors[t] +
    ";font-size:13px'>&#9632;</span> " + labels[t] +
    ": <b>" + fmt(yv) + "</b> <span style='opacity:.55'>@" + xs[i] + "</span></div>"
  );
}
const header = "<div style='font-weight:600;margin-bottom:3px'>step &asymp; " +
               Math.round(x) + "</div>";
readout.text = header + (rows.length ? rows.join("") :
               "<div style='opacity:.6'>(no visible traces here)</div>");
"""


def build_readout_callback(readout: Div, sources, labels, colors) -> CustomJS:
    """CustomJS that fills ``readout`` with every visible trace's value at the
    hovered step. Attach to each figure with ``fig.js_on_event('mousemove', cb)``."""
    return CustomJS(
        args=dict(readout=readout, srcs=list(sources), labels=list(labels),
                  colors=list(colors)),
        code=_READOUT_JS,
    )
