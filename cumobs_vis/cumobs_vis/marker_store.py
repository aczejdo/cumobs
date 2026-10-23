"""markerStore — event markers with CRUD + JSON persistence (plan §4h).

Each marker is a vertical line with a ``label``, a ``step``, and a ``category``
(``curriculum`` / ``detected`` / ``manual``) rendered with a **distinct style per
category**. Markers import/export as JSON so they persist across sessions and can
be reused across runs. Auto-detected events (a future extension) would live in a
separate layer; nothing here auto-detects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

CATEGORIES = ("curriculum", "detected", "manual")

# category -> Bokeh Span style (distinct per category).
CATEGORY_STYLES: dict[str, dict[str, object]] = {
    "curriculum": {"line_color": "#4f46e5", "line_dash": "solid", "line_width": 2},
    "detected":   {"line_color": "#ea580c", "line_dash": "dashed", "line_width": 2},
    "manual":     {"line_color": "#0d9488", "line_dash": "dotted", "line_width": 2},
}
_DEFAULT_STYLE = {"line_color": "#334155", "line_dash": "dashdot", "line_width": 1}


def style_for(category: str) -> dict[str, object]:
    return CATEGORY_STYLES.get(category, _DEFAULT_STYLE)


@dataclass
class Marker:
    id: int
    label: str
    step: float
    category: str = "manual"


class MarkerStore:
    def __init__(self) -> None:
        self._markers: dict[int, Marker] = {}
        self._next_id = 1

    # -- CRUD --------------------------------------------------------------

    def add(self, label: str, step: float, category: str = "manual") -> Marker:
        m = Marker(id=self._next_id, label=str(label), step=float(step),
                   category=category if category in CATEGORIES else "manual")
        self._markers[m.id] = m
        self._next_id += 1
        return m

    def update(self, marker_id: int, *, label: str | None = None,
               step: float | None = None, category: str | None = None) -> Marker | None:
        m = self._markers.get(marker_id)
        if m is None:
            return None
        if label is not None:
            m.label = str(label)
        if step is not None:
            m.step = float(step)
        if category is not None and category in CATEGORIES:
            m.category = category
        return m

    def delete(self, marker_id: int) -> bool:
        return self._markers.pop(marker_id, None) is not None

    def clear(self) -> None:
        self._markers.clear()

    def all(self) -> list[Marker]:
        return sorted(self._markers.values(), key=lambda m: (m.step, m.id))

    def __len__(self) -> int:
        return len(self._markers)

    # -- JSON persistence --------------------------------------------------

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps({"markers": [asdict(m) for m in self.all()]}, indent=indent)

    def load_json(self, text: str, replace: bool = True) -> int:
        """Load markers from a JSON string. ``replace`` clears existing markers
        first; otherwise imported markers are appended with fresh ids. Returns
        the number of markers imported. Tolerant of a bare list or a wrapped
        ``{"markers": [...]}`` object."""
        data = json.loads(text)
        items = data.get("markers", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("marker JSON must be a list or {'markers': [...]}")
        if replace:
            self.clear()
        count = 0
        for it in items:
            if not isinstance(it, dict) or "step" not in it:
                continue
            self.add(
                label=it.get("label", ""),
                step=it["step"],
                category=it.get("category", "manual"),
            )
            count += 1
        return count

    def export_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def import_file(self, path: str, replace: bool = True) -> int:
        with open(path, "r", encoding="utf-8") as fh:
            return self.load_json(fh.read(), replace=replace)
